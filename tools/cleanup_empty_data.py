#!/usr/bin/env python3
"""
空のdata_*フォルダを削除するスクリプト
"""

import os
import shutil
import glob

def cleanup_empty_data_folders():
    """dataフォルダ内の空のdata_*フォルダを削除"""

    # dataフォルダが存在するかチェック
    if not os.path.exists('data'):
        print("dataフォルダが存在しません")
        return

    # data_*パターンのフォルダを検索
    data_folders = glob.glob('data/data_*')

    if not data_folders:
        print("削除対象のdata_*フォルダが見つかりません")
        return

    deleted_count = 0

    for folder in data_folders:
        if os.path.isdir(folder):
            try:
                # フォルダ内のファイル一覧を取得
                files = os.listdir(folder)

                # 削除対象の判定
                should_delete = False
                delete_reason = ""

                if len(files) == 0:
                    # 完全に空のフォルダ
                    should_delete = True
                    delete_reason = "空フォルダ"
                elif len(files) <= 2 and all(f.endswith(('.json', '.txt')) for f in files):
                    # manifest.json、meta.json等の最小限のファイルのみ
                    should_delete = True
                    delete_reason = f"記録データなし - {files}"
                elif len(files) == 1 and files[0] == 'images':
                    # imagesフォルダのみ存在する場合
                    images_path = os.path.join(folder, 'images')
                    if os.path.isdir(images_path):
                        image_files = os.listdir(images_path)
                        if len(image_files) == 0:
                            # imagesフォルダが空
                            should_delete = True
                            delete_reason = "imagesフォルダのみ（空）"
                        else:
                            print(f"保持: {folder} (images: {len(image_files)} files)")
                    else:
                        should_delete = True
                        delete_reason = "imagesファイルのみ（フォルダではない）"
                else:
                    # その他のファイルがある場合は詳細チェック
                    has_data_files = False
                    for file in files:
                        file_path = os.path.join(folder, file)
                        if file != 'images' and os.path.isfile(file_path):
                            # catalog、manifest以外のファイルがあるかチェック
                            if not file.endswith(('.json', '.txt')):
                                has_data_files = True
                                break
                            elif file.endswith('.catalog'):
                                has_data_files = True
                                break

                    if not has_data_files:
                        should_delete = True
                        delete_reason = f"実質的な記録データなし - {files}"
                    else:
                        print(f"保持: {folder} ({len(files)} files)")

                if should_delete:
                    shutil.rmtree(folder)
                    print(f"削除: {folder} ({delete_reason})")
                    deleted_count += 1

            except Exception as e:
                print(f"エラー: {folder} の削除に失敗 - {e}")

    print(f"\n削除完了: {deleted_count} フォルダを削除しました")

if __name__ == "__main__":
    print("空のdata_*フォルダを削除します...")
    cleanup_empty_data_folders()