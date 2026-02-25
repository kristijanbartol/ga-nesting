import os
import subprocess
from sys import platform


def parameterize(hyperparams_config={}):
    # Get absolute paths
    current_dir = os.getcwd()
    cpp_program_path = os.path.join(current_dir, "anisotropic-parameterization/build/loom")
    root_project_path_arg = os.path.abspath(current_dir)
    
    # Ensure the executable exists
    if not os.path.exists(cpp_program_path):
        raise FileNotFoundError(f"Executable not found at: {cpp_program_path}")
    
    # Start with base command and config file
    command = [
        cpp_program_path,
        "--config", "anisotropic-parameterization/configs/default.json"
    ]
    
    # Add other parameters with their values
    param_mapping = {
        "matching_mode": "--matching-mode",
        "seamline_strategy": "--seamline-strategy",
        "num_seam_iters": "--num-seam-iters",
        "num_inner_iters": "--num-inner-iters",
        "max_stretch": "--max-stretch",
        "material_stretch_coef": "--material-stretch-coef",
        "stretch_coef": "--stretch-coef",
        "edges_coef": "--edges-coef",
        "seams_coef": "--seams-coef",
        "dart_coef": "--dart-coef"
    }

    for python_param, cpp_param in param_mapping.items():
        if python_param in hyperparams_config:
            value = hyperparams_config[python_param]
            if value is not None:  # Only add if value is specified
                command.extend([cpp_param, str(value)])
    
    print(f"Running command: {' '.join(command)}")
    
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=False,
            text=True,
            timeout=30,
            env=os.environ.copy()
        )
        print(f"Program output: {result.stdout}")
        return result
    except subprocess.TimeoutExpired as e:
        e.process.kill()
        e.process.wait()
        raise RuntimeError(
            f"Parameterization binary timed out after {e.timeout}s — "
            "likely stuck on a degenerate geometry configuration."
        )
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the C++ program: {e}")
        print(f"Program output: {e.stdout}")
        print(f"Error output: {e.stderr}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
