import datetime
import winreg
import os

def convert_unix_style_to_windows(path_str):
    if path_str.startswith("/"):
        parts = path_str.lstrip("/").split("/")
        if parts:
            drive = parts[0].upper() + ":"
            rest = parts[1:]
            return drive + '\\' + '\\'.join(rest)
    return path_str

def get_recent_pdf_files_from_tDIText(today,days):
    pdf_paths = []
    base_key_path = r"Software\Adobe\Adobe Acrobat\DC\AVGeneral\cRecentFiles"
    
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, base_key_path) as base_key:
            i = 0
            while True:
                try:
                    # サブキー名（例: "c1", "c2", ...）を取得
                    subkey_name = winreg.EnumKey(base_key, i)
                    i += 1

                    # サブキーを開く
                    with winreg.OpenKey(base_key, subkey_name) as subkey:
                        try:
                            # tDIText 値を取得（PDFファイルのパス）
                            value, _ = winreg.QueryValueEx(subkey, "tDIText")
                            if value.lower().endswith(".pdf"):
                                value1 = convert_unix_style_to_windows(value)
                                access_time = datetime.datetime.fromtimestamp(os.path.getatime(value1))
                                print(access_time)
                                # if (today - access_time.date()).days <= days:
                                pdf_paths.append(value1)
                        except FileNotFoundError:
                            pass  # tDITextがないサブキーは無視
                except OSError:
                    break  # サブキーがなくなったら終了

    except FileNotFoundError:
        print("Acrobatの最近のファイル履歴が見つかりません")

    return pdf_paths
days = 1
today = datetime.datetime.now().date()
recent_files = []
recent_files.extend(get_recent_pdf_files_from_tDIText(today,days))

print(len(recent_files))