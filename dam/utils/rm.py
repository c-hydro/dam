import os

def remove_file(file_path):
    """
    Remove a file from the filesystem.
    If the file was the last file in a directory, remove the directory as well.
    If the directory was the last directory in a path, remove the parent directory as well. etc.
    """

    os.remove(file_path)
    dir_path = os.path.dirname(file_path)
    while not os.listdir(dir_path):
        os.rmdir(dir_path)
        dir_path = os.path.dirname(dir_path)
        if not dir_path:
            break