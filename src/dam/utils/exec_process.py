import os
import subprocess

# -------------------------------------------------------------------------------------
# Method to execute process
def exec_process(command_line=None, command_path=None):

    try:

        # Execute command-line
        if command_path is not None:
            os.chdir(command_path)
        process_handle = subprocess.Popen(
            command_line, shell=True,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Read standard output
        while True:
            string_out = process_handle.stdout.readline()
            if isinstance(string_out, bytes):
                string_out = string_out.decode('UTF-8')

            if string_out == '' and process_handle.poll() is not None:

                if process_handle.poll() == 0:
                    break
                else:
                    raise RuntimeError('Error in executing process')

        # Collect stdout and stderr and exitcode
        std_out, std_err = process_handle.communicate()
        std_exit = process_handle.poll()

        if std_out == b'' or std_out == '':
            std_out = None
        if std_err == b'' or std_err == '':
            std_err = None

        # Check stream process
        stream_process(std_out, std_err)

        # Info command-line end
        return std_out, std_err, std_exit

    except subprocess.CalledProcessError:
        # Exit code for process error
        raise RuntimeError('Errors in the called executable')

    except OSError:
        # Exit code for os error
        raise RuntimeError('Executable not found!')

# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Method to stream process
def stream_process(std_out=None, std_err=None):

    if std_out is None and std_err is None:
        return True
    else:
        return False
# -------------------------------------------------------------------------------------