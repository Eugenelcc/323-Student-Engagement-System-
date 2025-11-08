"""
inspect_checkpoint.py

Usage (run in the same virtualenv you used to train / save the model):
  python scripts/inspect_checkpoint.py models/hybrid_miniX_mobilenet.pt

This helper will attempt to torch.load the checkpoint and describe whether it
contains a pickled Module or a state_dict. It prints actionable advice if
unpickling fails due to missing class definitions.
"""
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_checkpoint.py PATH_TO_PT")
        return 2
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return 2
    try:
        import torch
    except Exception as e:
        print("torch is not importable in this environment. Activate the venv used to train the model and try again.")
        print("Error:", e)
        return 2

    try:
        obj = torch.load(path, map_location="cpu")
    except Exception as e:
        # Print the exception and give guidance
        print("Failed to torch.load the checkpoint. Exception:\n", repr(e))
        if isinstance(e, AttributeError) or "Can't get attribute" in repr(e) or "module'" in repr(e):
            print("\nThis usually means the checkpoint was saved from a script where the model class was defined in __main__\n"
                  "or from a module that isn't available on PYTHONPATH when loading.\n\n"
                  "Fixes:\n"
                  " 1) Load the checkpoint in the same environment / project where the model class is defined.\n"
                  " 2) Ensure the original Python file with the class is on PYTHONPATH (or copy it into this project).\n"
                  " 3) If you can run the training code, re-save the model as a state_dict:\n"
                  "       torch.save(model.state_dict(), 'model_state.pt')\n"
                  "    Then load it here by instantiating the same model architecture and calling load_state_dict().\n"
                  " 4) Export to ONNX in the original environment for portability.\n")
        return 1

    # If we get here, torch.load succeeded
    import types
    import torch.nn as nn

    print(f"Loaded object type: {type(obj)}")
    if isinstance(obj, nn.Module):
        cls = obj.__class__
        print(f"Checkpoint contains a pickled nn.Module instance: {cls.__name__} (module={cls.__module__})")
        print("Recommendation: Ensure that the module path and class are available when loading in a different environment.\n"
              "If you want portability, re-save state_dict or export to ONNX.")
        return 0
    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"Checkpoint is a dict containing {len(keys)} keys.")
        # Common keys: state_dict, model, module, epoch, optimizer, etc.
        sample_keys = keys[:40]
        print("Top keys:", sample_keys)
        # If this dict looks like a state_dict (tensors), try to detect
        is_state_dict = all(k.endswith('weight') or k.endswith('bias') or isinstance(obj[k], torch.Tensor) for k in sample_keys)
        if is_state_dict or any('state_dict' in k for k in keys) or 'state_dict' in obj:
            # Try to detect where the real state dict is nested
            if 'state_dict' in obj and isinstance(obj['state_dict'], dict):
                sd = obj['state_dict']
            elif 'model' in obj and isinstance(obj['model'], dict):
                sd = obj['model']
            else:
                # Heuristic: assume the dict itself is a state_dict if values are tensors
                sd = obj
            print(f"Detected a state_dict with ~{len(sd)} parameter tensors.")
            # Print a few entries and shapes
            print("Sample parameters (name -> shape):")
            for i, (k, v) in enumerate(sd.items()):
                if i >= 20:
                    break
                try:
                    print(f"  {k} -> {tuple(v.shape)}")
                except Exception:
                    print(f"  {k} -> type={type(v)}")
            print("\nRecommendation: instantiate the same model architecture in code and call model.load_state_dict(...)\n"
                  "or re-save the full model object in the environment that created it (but note pickled objects require the original class to be importable).\n"
                  "To create an ONNX export, load the model in the original environment and run torch.onnx.export(...).")
            return 0
        else:
            print("The dict does not look like a plain state_dict. It may contain nested information (optimizer, training metadata, etc.).\n"
                  "If you see a key named 'model' or 'state_dict' above, check those entries as they often contain the weights.")
            return 0

    print("Unknown checkpoint content. Inspect manually in the environment that created the file.")
    return 0

if __name__ == '__main__':
    sys.exit(main())
