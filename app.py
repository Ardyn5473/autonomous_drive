from flask import Flask, render_template, request, redirect, url_for, send_from_directory, send_file, jsonify
import os
import pandas as pd
import uuid
import base64
import subprocess
import tempfile

app = Flask(__name__)

RECORDS_DIR = "records"
IMAGES_DIR = "images"

# CSVメタデータキャッシュ: {csvfile: {filename or "no_image_...": {各列}}}
metadata_cache = {}

def load_metadata_from_csv(csv_path):
    """
    CSVを読み込み、{キー: 行データ} の辞書を作成。
    image_file があれば basename(image_file) をキーに、
    なければ "no_image_行番号" をキーにして row_data["image_file"] = None とする。
    """
    df = pd.read_csv(csv_path)
    meta_dict = {}
    has_image_col = ("image_file" in df.columns)

    for i, row in df.iterrows():
        row_dict = row.to_dict()
        if has_image_col:
            img_path = row.get("image_file", "")
            # pd.isna()でNaNチェック
            if isinstance(img_path, float) and pd.isna(img_path):
                img_path = ""
        else:
            img_path = ""

        if img_path:
            fname = os.path.basename(str(img_path))
            meta_dict[fname] = row_dict
        else:
            key = f"no_image_{i}"
            row_dict["image_file"] = None
            meta_dict[key] = row_dict

    return meta_dict

@app.route("/", methods=["GET","POST"])
def slider():
    """
    1) POST: CSV選択 => ?csvfile=xxx にリダイレクト
    2) GET: CSVを読み込み、slider.html を表示
    """
    csv_files = [f for f in os.listdir(RECORDS_DIR) if f.endswith(".csv")]
    csv_files.sort()

    if request.method == "POST":
        selected_csv = request.form.get("selected_csv")
        if selected_csv:
            return redirect(url_for("slider", csvfile=selected_csv))
        else:
            # CSV未選択
            return render_template("slider.html",
                                   csv_files=csv_files,
                                   current_csv=None,
                                   image_files=[],
                                   metadata={})

    # GET
    csvfile = request.args.get("csvfile")
    if csvfile:
        csv_path = os.path.join(RECORDS_DIR, csvfile)
        if os.path.exists(csv_path):
            if csvfile not in metadata_cache:
                md = load_metadata_from_csv(csv_path)
                metadata_cache[csvfile] = md
            metadata = metadata_cache[csvfile]
            image_files = sorted(metadata.keys())
            current_csv = csvfile
        else:
            metadata = {}
            image_files = []
            current_csv = None
    else:
        metadata = {}
        image_files = []
        current_csv = None

    return render_template("slider.html",
                           csv_files=csv_files,
                           current_csv=current_csv,
                           image_files=image_files,
                           metadata=metadata)

@app.route("/images/<filename>")
def serve_raw_image(filename):
    """
    画像ファイルをそのまま返す。'no_image_...' はファイルが無いので404などにする。
    """
    if filename.startswith("no_image_"):
        return "No image", 404
    path = os.path.join(IMAGES_DIR, filename)
    if not os.path.exists(path):
        return f"File not found: {filename}", 404
    return send_from_directory(IMAGES_DIR, filename)

# -----------------------
# 削除機能: 単一, 範囲
# -----------------------

@app.route("/delete_range", methods=["POST"])
def delete_range():
    start_idx = request.form.get("start_index")
    end_idx = request.form.get("end_index")
    csvfile = request.form.get("csvfile", None)

    if not (start_idx and end_idx and csvfile):
        return "Missing parameters", 400

    try:
        start_idx = int(start_idx)
        end_idx = int(end_idx)
    except ValueError:
        return "Invalid index range", 400

    csv_path = os.path.join(RECORDS_DIR, csvfile)
    if not os.path.exists(csv_path):
        return f"CSV not found: {csvfile}", 404

    if csvfile not in metadata_cache:
        metadata_cache[csvfile] = load_metadata_from_csv(csv_path)
    md = metadata_cache[csvfile]
    image_files = sorted(md.keys())

    if start_idx < 0: start_idx = 0
    if end_idx > len(image_files)-1: end_idx = len(image_files)-1
    if start_idx > end_idx:
        return "Start index > End index", 400

    to_delete = image_files[start_idx:end_idx+1]
    _delete_files_and_csv(csvfile, to_delete)
    return redirect(url_for("slider", csvfile=csvfile))

@app.route("/delete_current", methods=["POST"])
def delete_current():
    csvfile = request.form.get("csvfile", None)
    idx_str = request.form.get("current_index", None)

    if not csvfile or idx_str is None:
        return "Missing parameters", 400

    try:
        idx = int(idx_str)
    except ValueError:
        return "Invalid current_index", 400

    csv_path = os.path.join(RECORDS_DIR, csvfile)
    if not os.path.exists(csv_path):
        return f"CSV not found: {csvfile}", 404

    if csvfile not in metadata_cache:
        metadata_cache[csvfile] = load_metadata_from_csv(csv_path)
    md = metadata_cache[csvfile]
    image_files = sorted(md.keys())

    if idx < 0 or idx >= len(image_files):
        return f"Invalid index {idx}", 400

    fname = image_files[idx]
    _delete_files_and_csv(csvfile, [fname])
    return redirect(url_for("slider", csvfile=csvfile))

def _delete_files_and_csv(csvfile, to_delete_fnames):
    csv_path = os.path.join(RECORDS_DIR, csvfile)
    md = metadata_cache[csvfile]

    for f in to_delete_fnames:
        if not f.startswith("no_image_"):
            path = os.path.join(IMAGES_DIR, f)
            if os.path.exists(path):
                os.remove(path)
        if f in md:
            md.pop(f)

    import pandas as pd
    df = pd.read_csv(csv_path)
    if "image_file" in df.columns:
        for f in to_delete_fnames:
            if f.startswith("no_image_"):
                df = df[~df["image_file"].isna()]
            else:
                fullpath = os.path.join(IMAGES_DIR, f)
                df = df[df["image_file"] != fullpath]
        df.to_csv(csv_path, index=False)

# -----------------------
# 動画作成: start~endをCanvas描画したフレームをアップロード
# -----------------------

frame_storage = {}  # { session_id: [list_of_frame_paths] }

@app.route("/start_recording", methods=["POST"])
def start_recording():
    """
    session_id 発行して空のフレームリストを用意
    """
    session_id = str(uuid.uuid4())
    frame_storage[session_id] = []
    return jsonify({"session_id": session_id})

@app.route("/upload_frame", methods=["POST"])
def upload_frame():
    """
    1フレームを受け取り、一時ファイルに保存
    """
    session_id = request.form.get("session_id")
    data_url = request.form.get("frame_data")
    if not session_id or not data_url:
        return "Missing session_id or frame_data", 400

    if session_id not in frame_storage:
        return "Invalid session_id", 400

    # "data:image/png;base64,..." -> base64
    header, encoded = data_url.split(",", 1)
    import base64
    frame_bin = base64.b64decode(encoded)

    import tempfile
    import uuid
    frame_file = f"{uuid.uuid4()}.png"
    temp_dir = tempfile.gettempdir()
    frame_path = os.path.join(temp_dir, frame_file)
    with open(frame_path, "wb") as f:
        f.write(frame_bin)

    frame_storage[session_id].append(frame_path)
    return "OK"

@app.route("/finish_recording", methods=["POST"])
def finish_recording():
    """
    受け取ったフレームを ffmpeg で動画化 & ダウンロードさせる
    """
    session_id = request.form.get("session_id")
    fps_str = request.form.get("fps", "10")
    try:
        fps = float(fps_str)
    except ValueError:
        fps = 10

    if not session_id or session_id not in frame_storage:
        return "Invalid session_id", 400

    frames = frame_storage[session_id]
    if not frames:
        return "No frames recorded", 400

    import tempfile
    import os
    import subprocess

    out_video = f"record_{session_id}.mp4"
    out_path = os.path.join(tempfile.gettempdir(), out_video)

    seqdir = os.path.join(tempfile.gettempdir(), session_id)
    os.makedirs(seqdir, exist_ok=True)

    # rename frames to 0001.png, 0002.png, ...
    for i, fpath in enumerate(frames):
        newname = os.path.join(seqdir, f"{i:04d}.png")
        os.rename(fpath, newname)

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", os.path.join(seqdir, "%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        out_path
    ]
    subprocess.run(cmd, check=True)

    # cleanup
    for f in os.listdir(seqdir):
        os.remove(os.path.join(seqdir, f))
    os.rmdir(seqdir)
    frame_storage.pop(session_id)  # remove from memory

    # mp4をダウンロード
    return send_file(out_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
