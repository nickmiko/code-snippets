# src/main.py
import os
import re
import time

def log_message(message, log_file):
    print(message)
    with open(log_file, "a") as f:
        f.write(message + "\n")

def detect_category(filename, ext):
    documents_ext = {
        'pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'txt', 'rtf', 'odt', 'ods', 'odp', 'csv', 'tsv',
        'tex', 'epub', 'mobi', 'djvu', 'ps', 'pages', 'numbers', 'key', 'md', 'rst', 'log', 'xml', 'json',
        'yml', 'yaml', 'ini', 'conf', 'cfg', 'sql', 'db', 'dbf', 'accdb', 'pub', 'wps', 'xps', 'sxw', 'sxc',
        'sxg', 'sxi', 'sxm', 'sxw', 'dot', 'dotx', 'docm', 'xlsm', 'pptm', 'pot', 'potx', 'pps', 'ppsx',
        'wpd', 'msg', 'eml', 'ics', 'vcf', 'tex', 'pdfa', 'pdfx', 'odf', 'fodt', 'fods', 'fodp', 'rtfd',
        'scriv', 'gdoc', 'gsheet', 'gslides', 'gdraw', 'gtable', 'gform', 'gmap', 'gscript', 'pages'
    }
    photo_ext = {
        'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'tif', 'heic', 'raw', 'cr2', 'nef', 'orf', 'sr2', 'arw',
        'dng', 'rw2', 'pef', 'raf', '3fr', 'erf', 'kdc', 'mos', 'nrw', 'srw', 'x3f', 'mef', 'psd', 'ai',
        'eps', 'svg', 'ico', 'jfif', 'webp', 'indd', 'avif', 'jpe', 'jp2', 'j2k', 'jpf', 'jpx', 'jpm',
        'pgm', 'ppm', 'pbm', 'xbm', 'icns', 'dds', 'exr', 'hdr', 'pic', 'pct', 'sgi', 'ras', 'im', 'mng',
        'apng', 'dib', 'emf', 'wmf', 'cur', 'fpx', 'pcd', 'pcx', 'pict', 'tga', 'vicar', 'xwd'
    }
    video_ext = {
        'mp4', 'mov', 'avi', 'mkv', 'wmv', 'flv', 'webm', 'mpeg', 'mpg', 'mpe', 'mp2', 'mpv',
        '3gp', '3g2', 'mts', 'm2ts', 'ts', 'vob', 'm4v', 'f4v', 'f4p', 'f4a', 'f4b', 'rm', 'rmvb',
        'divx', 'xvid', 'ogv', 'asf', 'amv', 'mxf', 'roq', 'svi', 'yuv', 'viv', 'qt', 'dv', 'dat',
        'h264', 'h265', 'hevc', 'avchd', 'vp9', 'vp8', 'mjpg', 'mj2'
    }
    music_ext = {
        'mp3', 'wav', 'aac', 'flac', 'ogg', 'm4a', 'wma', 'alac', 'aiff', 'ape', 'au', 'mp2', 'mp1', 'mpa',
        'opus', 'dsf', 'dff', 'wv', 'tta', 'ac3', 'amr', 'caf', 'snd', 'mid', 'midi', 'kar', 'rmi', 'gsm',
        'spx', 'ra', 'rm', 'mpc', 'shn', 'vqf', '3gp', '3g2', 'awb', 'dvf', 'msv', 'vox', 'raw', 'pcm',
        'oga', 'mogg', 'mod', 'it', 'xm', 's3m', 'mtm', 'stm', 'far', 'ult', '669', 'amf', 'ptm', 'med',
        'emod', 'umx', 'psm', 'j2b', 'mo3', 'xmf', 'rmi', 'hmp', 'mus', 'iff', 'sf2', 'sfz', 'sbk', 'sb0',
        'sb1', 'sb2', 'sb3', 'sb4', 'sb5', 'sb6', 'sb7', 'sb8', 'sb9', 'sb10', 'sb11', 'sb12', 'sb13',
        'sb14', 'sb15', 'sb16', 'sb17', 'sb18', 'sb19', 'sb20', 'sb21', 'sb22', 'sb23', 'sb24', 'sb25',
        'sb26', 'sb27', 'sb28', 'sb29', 'sb30', 'sb31', 'sb32', 'sb33', 'sb34', 'sb35', 'sb36', 'sb37',
        'sb38', 'sb39', 'sb40', 'sb41', 'sb42', 'sb43', 'sb44', 'sb45', 'sb46', 'sb47', 'sb48', 'sb49',
        'sb50', 'sb51', 'sb52', 'sb53', 'sb54', 'sb55', 'sb56', 'sb57', 'sb58', 'sb59', 'sb60', 'sb61',
        'sb62', 'sb63', 'sb64', 'sb65', 'sb66', 'sb67', 'sb68', 'sb69', 'sb70', 'sb71', 'sb72', 'sb73',
        'sb74', 'sb75', 'sb76', 'sb77', 'sb78', 'sb79', 'sb80', 'sb81', 'sb82', 'sb83', 'sb84', 'sb85',
        'sb86', 'sb87', 'sb88', 'sb89', 'sb90', 'sb91', 'sb92', 'sb93', 'sb94', 'sb95', 'sb96', 'sb97',
        'sb98', 'sb99'
    }
    ext = ext.lower()
    if ext in documents_ext:
        return "Documents"
    elif ext in photo_ext:
        return "Photo"
    elif ext in video_ext:
        return "Video"
    elif ext in music_ext:
        return "Music"
    else:
        return "Other"

def detect_year(filename, file_path):
    # Try to find a year in the filename (e.g., 2020, 2019, etc.)
    match = re.search(r'(19|20)\d{2}', filename)
    if match:
        year = match.group(0)
        # Ignore current year as "too recent"
        current_year = str(time.localtime().tm_year)
        if year != current_year:
            return year
    # If not found, try file's modification time
    try:
        mod_time = os.path.getmtime(file_path)
        year = time.strftime('%Y', time.localtime(mod_time))
        current_year = str(time.localtime().tm_year)
        if year != current_year:
            return year
    except Exception:
        pass
    return None

def organize_files(root_dir, log_file):
    """
    Scans a directory and its subdirectories, organizing files by type.

    Args:
        root_dir (str): The root directory to start the scan from.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip extension folders in root to avoid moving files twice
        if dirpath == root_dir:
            skip_dirs = {os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))}
        else:
            skip_dirs = set()
        for filename in filenames:
            if filename.startswith('.'):
                continue  # skip hidden files
            file_ext = filename.split('.')[-1]
            file_path = os.path.join(dirpath, filename)
            category = detect_category(filename, file_ext)
            year = detect_year(filename, file_path)
            # Build target dir
            target_dir = os.path.join(root_dir, category)
            if year:
                target_dir = os.path.join(target_dir, year)
            duplicates_dir = os.path.join(target_dir, "duplicates")
            # Skip if already in target_dir or duplicates_dir
            if os.path.abspath(dirpath) == os.path.abspath(target_dir) or os.path.abspath(dirpath) == os.path.abspath(duplicates_dir):
                continue
            # If a file exists where the target_dir should be, move it to skipped_files and skip
            if os.path.isfile(target_dir):
                skipped_dir = os.path.join(root_dir, "skipped_files")
                os.makedirs(skipped_dir, exist_ok=True)
                skipped_file_path = os.path.join(skipped_dir, os.path.basename(target_dir))
                try:
                    os.rename(target_dir, skipped_file_path)
                    log_message(f"Warning: {target_dir} exists as a file. Moved to {skipped_file_path}. Skipping category '{category}'.", log_file)
                except Exception as e:
                    log_message(f"Error moving file {target_dir} to skipped_files: {e}", log_file)
                continue
            os.makedirs(target_dir, exist_ok=True)
            target_path = os.path.join(target_dir, filename)
            try:
                if os.path.exists(target_path):
                    log_message(f"Duplicate found: {filename} in {target_dir}. Moving both to duplicates folder.", log_file)
                    os.makedirs(duplicates_dir, exist_ok=True)
                    existing_file_in_duplicates = os.path.join(duplicates_dir, filename)
                    if not os.path.exists(existing_file_in_duplicates):
                        os.rename(target_path, existing_file_in_duplicates)
                        log_message(f"Moved existing file to {existing_file_in_duplicates}", log_file)
                    base, ext = os.path.splitext(filename)
                    count = 1
                    new_filename = filename
                    while os.path.exists(os.path.join(duplicates_dir, new_filename)):
                        new_filename = f"{base}_{count}{ext}"
                        count += 1
                    os.rename(file_path, os.path.join(duplicates_dir, new_filename))
                    log_message(f"Moved duplicate file to {os.path.join(duplicates_dir, new_filename)}", log_file)
                else:
                    os.rename(file_path, target_path)
                    log_message(f"Moved {filename} to {target_dir}", log_file)
            except Exception as e:
                log_message(f"Error moving {filename}: {e}", log_file)

def move_empty_folders(root_dir, log_file):
    old_folders_dir = os.path.join(root_dir, "old_pre_move_folders")
    os.makedirs(old_folders_dir, exist_ok=True)
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        # Skip root and old_pre_move_folders itself
        if dirpath in [root_dir, old_folders_dir]:
            continue
        # Move any empty folder
        if not dirnames and not filenames:
            base = os.path.basename(dirpath)
            target = os.path.join(old_folders_dir, base)
            try:
                os.rename(dirpath, target)
                log_message(f"Moved empty folder {dirpath} to {target}", log_file)
            except Exception as e:
                log_message(f"Error moving empty folder {dirpath}: {e}", log_file)

if __name__ == "__main__":
    root_directory = input("Enter the root directory to scan: ")
    if not os.path.isdir(root_directory):
        print(f"Error: Provided path is not a directory.\nReceived: '{root_directory}'")
        print("Tip: Check the path and ensure it exists. If using spaces, enclose the path in quotes.")
    else:
        folder_name = os.path.basename(os.path.normpath(root_directory))
        log_file = os.path.join(root_directory, f"{folder_name}_organizer.log")
        log_message(f"Organizing files in {root_directory}...", log_file)
        organize_files(root_directory, log_file)
        move_empty_folders(root_directory, log_file)
        log_message("File organization complete.", log_file)
        log_message("Appended to log, don't erase", log_file)