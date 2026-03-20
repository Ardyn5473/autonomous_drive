from flask import Flask, render_template, jsonify, request, send_file, redirect, Response
from flask_cors import CORS
import os
import json
import glob
import logging
import base64
import platform
import numpy as np
import sys
import subprocess
import webbrowser
import threading
import inspect
from pathlib import Path
from data_loader import DonkeycarDataLoader
from training_manager import TrainingManager
from neural_network import TrainingConfig

app = Flask(__name__)
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize data loader
data_loader = DonkeycarDataLoader()

# Initialize training manager
training_manager = TrainingManager()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_file(os.path.join(app.static_folder, 'favicon.ico'), mimetype='image/x-icon')

@app.route('/api/browse', methods=['GET'])
def browse_directory():
    """Browse directory structure"""
    current_path = request.args.get('path', os.path.expanduser('~'))
    try:
        # Ensure path is absolute and safe
        current_path = os.path.abspath(current_path)
        
        items = []
        
        # Add parent directory option (except for root)
        parent_path = os.path.dirname(current_path)
        if parent_path != current_path:  # Not at root
            items.append({
                'name': '..',
                'path': parent_path,
                'type': 'parent',
                'is_data_folder': False
            })
        
        # Check if current directory itself is a data folder
        current_has_catalog = any(f.endswith('.catalog') for f in os.listdir(current_path) if os.path.isfile(os.path.join(current_path, f)))

        # List directory contents
        for item in sorted(os.listdir(current_path), key=lambda x: os.path.getmtime(os.path.join(current_path, x)), reverse=True):
            if item.startswith('.'):  # Skip hidden files
                continue

            item_path = os.path.join(current_path, item)

            if os.path.isdir(item_path):
                # Check if it's a data folder
                # Pattern 1: folder/data/*.catalog (Donkeycar format)
                # Pattern 2: folder/*.catalog (direct data folder)
                has_catalog = False

                # Check Pattern 1: subfolder named 'data' with catalog files
                data_path = os.path.join(item_path, 'data')
                if os.path.exists(data_path):
                    try:
                        files = os.listdir(data_path)
                        has_catalog = any(f.endswith('.catalog') for f in files)
                    except:
                        pass

                # Check Pattern 2: catalog files directly in folder
                if not has_catalog:
                    try:
                        files = os.listdir(item_path)
                        has_catalog = any(f.endswith('.catalog') for f in files if os.path.isfile(os.path.join(item_path, f)))
                    except:
                        pass

                items.append({
                    'name': item,
                    'path': item_path,
                    'type': 'directory',
                    'is_data_folder': has_catalog
                })
        
        return jsonify({
            'current_path': current_path,
            'items': items
        })
    except Exception as e:
        logger.exception('API error')
        return jsonify({'error': str(e)}), 500

@app.route('/api/folders', methods=['GET'])
def get_folders():
    """Get list of data folders (legacy endpoint)"""
    # Default to parent directory to find data folders
    base_path = request.args.get('path', '..')
    try:
        folders = []
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path) and item.startswith('data'):
                # Check if it contains data subfolder with catalog files
                data_path = os.path.join(item_path, 'data')
                if os.path.exists(data_path):
                    try:
                        files = os.listdir(data_path)
                        if any(f.endswith('.catalog') for f in files):
                            folders.append({
                                'name': item,
                                'path': os.path.abspath(item_path),  # Use absolute path
                                'has_data': True
                            })
                    except:
                        pass
        return jsonify({'folders': folders})
    except Exception as e:
        logger.exception('API error')
        return jsonify({'error': str(e)}), 500

@app.route('/api/load_data', methods=['POST'])
def load_data():
    """Load data from selected folder"""
    try:
        folder_path = request.json.get('folder_path')
        if not folder_path:
            return jsonify({'error': 'No folder path provided'}), 400
        
        data_loader.load_data(folder_path)
        
        # Get basic info about loaded data
        info = {
            'total_records': len(data_loader.records),
            'sessions': data_loader.get_sessions(),
            'data_keys': data_loader.get_data_keys(),
            'timestamp_range': data_loader.get_timestamp_range(),
            'deleted_indexes': data_loader.get_deleted_indexes()
        }
        
        # Add LiDAR metadata if available
        lidar_metadata = data_loader.get_lidar_metadata()
        if lidar_metadata:
            info['lidar_metadata'] = lidar_metadata

        return jsonify({'success': True, 'info': info})
    except Exception as e:
        logger.exception('API error')
        return jsonify({'error': str(e)}), 500

@app.route('/api/lidar/<path:npy_filename>')
def get_lidar(npy_filename):
    """Serve a single LiDAR .npy file as JSON distances"""
    try:
        distances = data_loader.load_lidar_data(npy_filename)
        if distances is None:
            return jsonify({'error': 'LiDAR data not found'}), 404

        response = jsonify({'distances': distances})
        response.headers['Cache-Control'] = 'public, max-age=86400, immutable'
        return response
    except Exception as e:
        logger.exception('API error')
        return jsonify({'error': str(e)}), 500

@app.route('/api/lidar_batch', methods=['POST'])
def get_lidar_batch():
    """Serve multiple LiDAR .npy files in a single response"""
    try:
        data = request.json
        filenames = data.get('filenames', [])

        if not filenames:
            return jsonify({'error': 'No filenames provided'}), 400

        lidar_data = {}
        for filename in filenames:
            distances = data_loader.load_lidar_data(filename)
            if distances is not None:
                lidar_data[filename] = distances

        return jsonify({'lidar_data': lidar_data})
    except Exception as e:
        logger.exception('API error')
        return jsonify({'error': str(e)}), 500

@app.route('/api/data', methods=['GET'])
def get_data():
    """Get data records with pagination"""
    try:
        # Get query parameters
        start_idx = int(request.args.get('start', 0))
        end_idx = request.args.get('end', None)
        if end_idx is not None:
            end_idx = int(end_idx)
        session_id = request.args.get('session', None)
        
        # Get filtered records
        records = data_loader.get_records(start_idx, end_idx, session_id)
        
        return jsonify({
            'records': records,
            'total': len(data_loader.records)
        })
    except Exception as e:
        logger.exception('API error')
        return jsonify({'error': str(e)}), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get statistics for numerical data"""
    try:
        key = request.args.get('key')
        session_id = request.args.get('session', None)
        
        if not key:
            # Return statistics for all numerical keys
            all_stats = {}
            for k in data_loader.get_numerical_keys():
                stats = data_loader.calculate_statistics(k, session_id)
                if stats:
                    all_stats[k] = stats
            return jsonify(all_stats)
        else:
            # Return statistics for specific key
            stats = data_loader.calculate_statistics(key, session_id)
            if stats:
                return jsonify({key: stats})
            else:
                return jsonify({'error': 'Key not found or not numerical'}), 404
    except Exception as e:
        logger.exception('API error')
        return jsonify({'error': str(e)}), 500

@app.route('/api/image/<path:image_path>')
def get_image(image_path):
    """Serve image files, optionally with reduced JPEG quality"""
    try:
        # Use data_path which points to the actual data location
        full_path = os.path.join(data_loader.data_path, 'images', image_path)

        if not os.path.exists(full_path):
            return jsonify({'error': 'Image not found'}), 404

        quality = request.args.get('quality', type=int)

        if quality and 1 <= quality < 95:
            from PIL import Image as PILImage
            import io
            img = PILImage.open(full_path)
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=quality, optimize=True)
            buf.seek(0)
            response = send_file(buf, mimetype='image/jpeg')
        else:
            response = send_file(full_path, mimetype='image/jpeg')

        response.headers['Cache-Control'] = 'public, max-age=86400, immutable'
        return response
    except Exception as e:
        logger.exception('API error')
        return jsonify({'error': str(e)}), 500

@app.route('/api/images_batch', methods=['POST'])
def get_images_batch():
    """Serve multiple images in a single response as base64-encoded JSON"""
    try:
        data = request.json
        paths = data.get('paths', [])
        quality = data.get('quality', None)

        if not paths:
            return jsonify({'error': 'No paths provided'}), 400

        results = {}
        images_dir = os.path.join(data_loader.data_path, 'images')

        for image_path in paths:
            full_path = os.path.join(images_dir, image_path)
            if os.path.exists(full_path):
                with open(full_path, 'rb') as f:
                    img_data = f.read()

                if quality and 1 <= quality < 95:
                    from PIL import Image as PILImage
                    import io
                    img = PILImage.open(io.BytesIO(img_data))
                    buf = io.BytesIO()
                    img.save(buf, format='JPEG', quality=quality, optimize=True)
                    img_data = buf.getvalue()

                results[image_path] = base64.b64encode(img_data).decode('ascii')

        return jsonify({'images': results})
    except Exception as e:
        logger.exception('API error')
        return jsonify({'error': str(e)}), 500

@app.route('/api/timeline', methods=['GET'])
def get_timeline():
    """Get timeline data for visualization"""
    try:
        session_id = request.args.get('session', None)
        key = request.args.get('key', 'user/throttle')

        timeline_data = data_loader.get_timeline_data(key, session_id)

        return jsonify({
            'key': key,
            'data': timeline_data
        })
    except Exception as e:
        logger.exception('API error')
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete_indexes', methods=['POST'])
def update_delete_indexes():
    """Update deleted indexes in manifest file"""
    try:
        data = request.json
        start_idx = data.get('start_idx')
        end_idx = data.get('end_idx')

        if start_idx is None or end_idx is None:
            return jsonify({'error': 'start_idx and end_idx are required'}), 400

        if start_idx > end_idx:
            return jsonify({'error': 'start_idx must be <= end_idx'}), 400

        deleted_indexes = data_loader.update_deleted_indexes(start_idx, end_idx)

        return jsonify({
            'success': True,
            'deleted_indexes': deleted_indexes,
            'count': len(deleted_indexes)
        })
    except Exception as e:
        logger.exception('API error')
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear_delete_indexes', methods=['POST'])
def clear_delete_indexes():
    """Clear deleted indexes in manifest file"""
    try:
        data = request.json
        start_idx = data.get('start_idx')
        end_idx = data.get('end_idx')

        if start_idx is None or end_idx is None:
            return jsonify({'error': 'start_idx and end_idx are required'}), 400

        if start_idx > end_idx:
            return jsonify({'error': 'start_idx must be <= end_idx'}), 400

        deleted_indexes = data_loader.clear_deleted_indexes(start_idx, end_idx)

        return jsonify({
            'success': True,
            'deleted_indexes': deleted_indexes,
            'count': len(deleted_indexes)
        })
    except Exception as e:
        logger.exception('API error')
        return jsonify({'error': str(e)}), 500


# =====================================================
# Training API Endpoints
# =====================================================

@app.route('/api/training/train', methods=['POST'])
def train_model():
    """Start new model training"""
    try:
        data = request.json
        config = TrainingConfig.from_dict(data)
        result = training_manager.train_model(config, data_loader.records, data_path=data_loader.data_path, continue_training=False)

        if 'error' in result:
            return jsonify(result), 400

        return jsonify(result)
    except Exception as e:
        logger.exception('API error')
        return jsonify({'error': str(e)}), 500


@app.route('/api/training/continue', methods=['POST'])
def continue_training():
    """Continue training from existing model"""
    try:
        data = request.json
        config = TrainingConfig.from_dict(data)
        result = training_manager.train_model(config, data_loader.records, data_path=data_loader.data_path, continue_training=True)

        if 'error' in result:
            return jsonify(result), 400

        return jsonify(result)
    except Exception as e:
        logger.exception('API error')
        return jsonify({'error': str(e)}), 500


@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """Stop current training"""
    try:
        result = training_manager.stop_training()
        return jsonify(result)
    except Exception as e:
        logger.exception('API error')
        return jsonify({'error': str(e)}), 500


@app.route('/api/training/progress', methods=['GET'])
def get_training_progress():
    """Get current training progress"""
    try:
        progress = training_manager.get_progress()
        return jsonify(progress)
    except Exception as e:
        logger.exception('API error')
        return jsonify({'error': str(e)}), 500


# =====================================================
# Model Management API Endpoints
# =====================================================

@app.route('/api/models', methods=['GET'])
def list_models():
    """List all saved models"""
    try:
        models = training_manager.list_models()
        models_dir = str(Path(training_manager.models_dir).resolve())
        return jsonify({'models': models, 'models_dir': models_dir})
    except Exception as e:
        logger.exception('API error')
        return jsonify({'error': str(e)}), 500

@app.route('/api/open_folder', methods=['POST'])
def open_folder():
    """Open a folder in the system file explorer"""
    try:
        folder_path = request.json.get('path', '')
        folder_path = os.path.abspath(folder_path)
        if not os.path.isdir(folder_path):
            return jsonify({'error': 'Folder not found'}), 404
        if platform.system() == 'Windows':
            os.startfile(folder_path)
        elif platform.system() == 'Darwin':
            subprocess.Popen(['open', folder_path])
        else:
            subprocess.Popen(['xdg-open', folder_path])
        return jsonify({'message': 'Folder opened', 'path': folder_path})
    except Exception as e:
        logger.exception('API error')
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/load', methods=['POST'])
def load_model():
    """Load a model from file"""
    try:
        data = request.json
        model_path = data.get('model_path')

        if not model_path:
            return jsonify({'error': 'model_path is required'}), 400

        result = training_manager.load_model(model_path)

        if 'error' in result:
            return jsonify(result), 400

        return jsonify(result)
    except Exception as e:
        logger.exception('API error')
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/delete', methods=['POST'])
def delete_model():
    """Delete a model file"""
    try:
        data = request.json
        model_path = data.get('model_path')

        if not model_path:
            return jsonify({'error': 'model_path is required'}), 400

        result = training_manager.delete_model(model_path)

        if 'error' in result:
            return jsonify(result), 400

        return jsonify(result)
    except Exception as e:
        logger.exception('API error')
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/predict', methods=['POST'])
def predict():
    """Run prediction with current model"""
    try:
        data = request.json
        deleted_indexes = data.get('deleted_indexes', [])

        result = training_manager.predict(data_loader.records, deleted_indexes, data_path=data_loader.data_path)

        if 'error' in result:
            return jsonify(result), 400

        return jsonify(result)
    except Exception as e:
        logger.exception('API error')
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/current', methods=['GET'])
def get_current_model():
    """Get current loaded model info"""
    try:
        model_info = training_manager.get_current_model_info()
        if model_info:
            return jsonify({'model_info': model_info})
        else:
            return jsonify({'model_info': None})
    except Exception as e:
        logger.exception('API error')
        return jsonify({'error': str(e)}), 500


# =====================================================
# MLflow API Endpoints
# =====================================================

@app.route('/api/mlflow/start', methods=['POST'])
def start_mlflow_ui():
    """Start MLflow UI server"""
    try:
        # Check if MLflow UI is already running
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 8011))
        sock.close()

        if result == 0:
            return jsonify({'message': 'MLflow UI already running', 'port': 8011})

        # Start MLflow UI on port 8011 (to avoid conflict with data_viewer on 8010)
        mlruns_path = Path(training_manager.mlruns_dir).resolve().as_uri()
        proc = subprocess.Popen(
            [sys.executable, '-m', 'mlflow', 'ui',
             '--host', '0.0.0.0', '--port', '8011',
             '--backend-store-uri', mlruns_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info(f'MLflow UI process started (pid={proc.pid}), mlruns={mlruns_path}')

        return jsonify({'message': 'MLflow UI started', 'port': 8011})
    except Exception as e:
        logger.exception('API error')
        return jsonify({'error': str(e)}), 500


@app.route('/api/platform', methods=['GET'])
def get_platform():
    """Detect platform (RPi vs desktop) for adaptive buffer sizing"""
    try:
        is_rpi = False
        model_name = ''

        # Check /proc/device-tree/model for Raspberry Pi
        device_tree_model = '/proc/device-tree/model'
        if os.path.exists(device_tree_model):
            try:
                with open(device_tree_model, 'r') as f:
                    model_name = f.read().strip('\x00').strip()
                if 'raspberry pi' in model_name.lower():
                    is_rpi = True
            except Exception:
                pass

        # Fallback: check ARM architecture
        if not is_rpi:
            machine = platform.machine().lower()
            if machine in ('aarch64', 'armv7l', 'armv6l'):
                is_rpi = True

        # Get total RAM from /proc/meminfo
        total_ram_mb = 0
        if os.path.exists('/proc/meminfo'):
            try:
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            parts = line.split()
                            total_ram_mb = int(parts[1]) // 1024
                            break
            except Exception:
                pass

        profile = 'rpi' if is_rpi else 'desktop'

        return jsonify({
            'profile': profile,
            'model': model_name,
            'arch': platform.machine(),
            'total_ram_mb': total_ram_mb
        })
    except Exception as e:
        logger.exception('API error')
        return jsonify({'error': str(e)}), 500


@app.route('/api/sensors', methods=['GET'])
def get_available_sensors():
    """Get list of available sensor keys from loaded data"""
    try:
        if not data_loader.records:
            return jsonify({'sensors': []})

        # Get first record to find sensor keys
        first_record = data_loader.records[0]

        # Find all keys that can be used as input features
        # Exclude internal keys, image keys, and output keys
        excluded_prefixes = ('_', 'cam', 'user/')
        excluded_keys = {'_index', '_absolute_index', '_display_index', '_is_deleted'}

        sensor_list = []
        for key, value in first_record.items():
            # Skip excluded keys
            if key in excluded_keys:
                continue
            if any(key.startswith(prefix) for prefix in excluded_prefixes):
                continue
            # Numerical values → scalar sensor
            if isinstance(value, (int, float)):
                sensor_list.append({'key': key, 'type': 'scalar', 'size': 1})
            # .npy string values (distance_array) → array sensor
            elif isinstance(value, str) and value.endswith('.npy') and key.endswith('/distance_array'):
                array_size = data_loader.get_lidar_array_size(value)
                if array_size is not None:
                    sensor_list.append({'key': key, 'type': 'array', 'size': array_size})

        # Preserve original order from catalog (don't sort)
        return jsonify({'sensors': sensor_list})
    except Exception as e:
        logger.exception('API error')
        return jsonify({'error': str(e)}), 500


@app.route('/api/code', methods=['GET'])
def get_code():
    """Get source code sections for display"""
    try:
        sections = []

        # Section 1: Model definition (neural_network.py full file)
        nn_path = os.path.join(os.path.dirname(__file__), 'neural_network.py')
        with open(nn_path, 'r') as f:
            nn_code = f.read()
        sections.append({'title': 'モデル定義 (neural_network.py)', 'code': nn_code})

        # Section 2: Training loop (_train_model_internal method)
        train_code = inspect.getsource(TrainingManager._train_model_internal)
        sections.append({'title': '学習ループ (_train_model_internal)', 'code': train_code})

        # Section 3: Data loader (_build_input_matrix method)
        dataloader_code = inspect.getsource(TrainingManager._build_input_matrix)
        sections.append({'title': 'データローダー (_build_input_matrix)', 'code': dataloader_code})

        # Section 4: Data sample from currently loaded records
        if data_loader.records:
            sample_lines = []
            sample_lines.append(f"# データパス: {data_loader.data_path}")
            lidar_path = os.path.join(data_loader.data_path, 'lidar')
            if os.path.isdir(lidar_path):
                sample_lines.append(f"# LiDARパス: {lidar_path}")
            sample_lines.append(f"# 読み込み済みデータ: 全 {len(data_loader.records)} レコード")
            sample_lines.append(f"# キー一覧: {list(data_loader.records[0].keys())}")
            sample_lines.append("")

            sample_count = min(3, len(data_loader.records))
            for i in range(sample_count):
                rec = data_loader.records[i]
                sample_lines.append(f"# --- レコード [{i}] ---")
                sample_lines.append(f"records[{i}] = {{")
                for key, value in rec.items():
                    if isinstance(value, str) and value.endswith('.npy') and key.endswith('/distance_array'):
                        # LiDAR: .npy ファイルを読み込んで実データを表示
                        lidar_data = data_loader.load_lidar_data(value)
                        if lidar_data is not None:
                            sample_lines.append(f"    {key!r}: {value!r},")
                            sample_lines.append(f"    # ↑ LiDAR点群 ({len(lidar_data)}点): 実データ ↓")
                            # 先頭・末尾を省略表示
                            arr_str = repr(lidar_data[:10])[:-1] + ', ..., ' + repr(lidar_data[-3:])[1:]
                            sample_lines.append(f"    # distances = {arr_str}")
                            sample_lines.append(f"    # min={min(lidar_data):.1f}, max={max(lidar_data):.1f}, mean={sum(lidar_data)/len(lidar_data):.1f}")
                        else:
                            sample_lines.append(f"    {key!r}: {value!r},  # (LiDAR .npy 読み込み不可)")
                    else:
                        sample_lines.append(f"    {key!r}: {value!r},")
                sample_lines.append("}")
                sample_lines.append("")

            sections.append({
                'title': f'データサンプル（先頭 {sample_count} 件 / 全 {len(data_loader.records)} 件）',
                'code': '\n'.join(sample_lines)
            })

        return jsonify({'sections': sections})
    except Exception as e:
        logger.exception('API error')
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # use_reloader=False to prevent watchdog from restarting during training
    # (PyTorch/MLflow imports trigger file changes that cause reloads)

    PORT = 8010

    # Open browser automatically after a short delay
    def open_browser():
        webbrowser.open(f'http://localhost:{PORT}')

    threading.Timer(1.5, open_browser).start()

    app.run(host='0.0.0.0', port=PORT, debug=True, use_reloader=False)