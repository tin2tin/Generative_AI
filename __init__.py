# https://modelscope.cn/models/damo/text-to-video-synthesis/summary

bl_info = {
    "name": "Text to Video",
    "author": "tintwotin",
    "version": (1, 0),
    "blender": (3, 4, 0),
    "location": "Video Sequence Editor > Sidebar > Text to Video",
    "description": "Convert text to video",
    "category": "Sequencer",
}

import bpy, ctypes
from bpy.types import Operator, Panel
import site
import subprocess
import sys, os


def show_system_console(show):
    # https://docs.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-showwindow
    SW_HIDE = 0
    SW_SHOW = 5

    ctypes.windll.user32.ShowWindow(
        ctypes.windll.kernel32.GetConsoleWindow(), SW_SHOW if show else SW_HIDE
    )


def set_system_console_topmost(top):
    # https://docs.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setwindowpos
    HWND_NOTOPMOST = -2
    HWND_TOPMOST = -1
    HWND_TOP = 0
    SWP_NOMOVE = 0x0002
    SWP_NOSIZE = 0x0001
    SWP_NOZORDER = 0x0004

    ctypes.windll.user32.SetWindowPos(
        ctypes.windll.kernel32.GetConsoleWindow(),
        HWND_TOP if top else HWND_NOTOPMOST,
        0,
        0,
        0,
        0,
        SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER,
    )


def import_module(self, module, install_module):
    show_system_console(True)
    set_system_console_topmost(True)   
    
    module = str(module)
    try:
        exec("import " + module)
    except ModuleNotFoundError:
        app_path = site.USER_SITE
        if app_path not in sys.path:
            sys.path.append(app_path)
        pybin = sys.executable
        self.report({"INFO"}, "Installing: " + module + " module.")
        print("Installing: " + module + " module")
        subprocess.check_call([pybin, "-m", "pip", "install", install_module, "--no-warn-script-location", "--user"])
        try:
            exec("import " + module)
        except ModuleNotFoundError:
            return False
    return True


class SequencerImportMovieOperator(Operator):
    """Text to Video"""

    bl_idname = "sequencer.import_movie"
    bl_label = "Prompt"
    bl_description = "Convert text to video"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        if not bpy.types.Scene.text_prompt:
            return {"CANCELLED"}
        scene = context.scene

        app_path = site.USER_SITE
        if app_path not in sys.path:
            sys.path.append(app_path)
        pybin = sys.executable

        print("Ensuring: pip")
        try:
            subprocess.call([pybin, "-m", "ensurepip"])
            subprocess.call([pybin, "-m", "pip", "install", "--upgrade","pip"])
        except ImportError:
            pass

        import_module(self, "open_clip_torch", "open_clip_torch")
        import_module(self, "pytorch_lightning", "pytorch_lightning")
        import_module(self, "addict", "addict")
        import_module(self, "yapf", "yapf")
        import_module(self, "datasets", "datasets")
        import_module(self, "einops", "einops")
        import_module(self, "jsonplus", "jsonplus") 
        import_module(self, "oss2", "oss2")
        import_module(self, "pyarrow", "pyarrow")
        import_module(self, "huggingface_hub", "--upgrade huggingface_hub")
        import_module(self, "numpy", "--upgrade numpy")
        import_module(self, "gast", "gast")
        import_module(self, "diffusers", "diffusers")
        import_module(self, "tensorflow", "tensorflow")
        #import_module(self, "torch", "--index-url https://download.pytorch.org/whl/cu118")
        subprocess.check_call([pybin, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cu200"])
        #torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        import_module(self, "diffusers", "git+https://github.com/huggingface/diffusers.git")
        import_module(self, "transformers", "git+https://github.com/huggingface/transformers.git")
        import_module(self, "accelerate", "accelerate")
        import_module(self, "opencv_python", "opencv_python")
        import_module(self, "modelscope", "modelscope==1.4.2") #git+https://github.com/modelscope/modelscope.git

        from huggingface_hub import snapshot_download

        from modelscope.pipelines import pipeline
        from modelscope.outputs import OutputKeys
        import torch

        import gc
        #from pathlib import Path
        import pathlib

        import cv2
        import torch
        from diffusers import DiffusionPipeline

        script_file = os.path.realpath(__file__)
        directory = os.path.dirname(script_file)
        model_dir = os.path.join(directory, "model")
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        
        # list of file names to check
        files = ["configuration.json", "VQGAN_autoencoder.pth", "open_clip_pytorch_model.bin", "text2video_pytorch_model.pth"]

        all_found = True

        # loop over the files and check if they exist
        for filename in files:
            check_file = os.path.join(model_dir, filename)
            #print(check_file)
            check_file = pathlib.Path(check_file) 
            if not os.path.isfile(check_file):
                print(check_file)
                all_found = False                  

        if not all_found: #snapshot_download(repo_id='damo-vilab/modelscope-damo-text-to-video-synthesis',  # 20 GB VRAM
            snapshot_download(repo_id='kabachuha/modelscope-damo-text2video-pruned-weights', # 6GB VRAM
                              repo_type='model',
                              local_dir=model_dir,
                              local_dir_use_symlinks=False)


        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
#        torch.zeros(1).cpu()
#        # Run model
#        pipeline = make_pipeline_generator(
#            device=device, cpu_offload=cpu_offload, attention_slice=attention_slice
#        )
        generator = torch.Generator(device=torch.device(device))#.manual_seed(seed)
#        video = pipeline(
#            prompt=prompt,
#            num_frames=num_frames,
#            num_inference_steps=num_steps,
#            height=height,
#            width=width,
#            generator=generator,
#        ).frames
#kabachuha/modelscope-damo-text2video-pruned-weights

        p = pipeline( 'text-to-video-synthesis', model_dir, #'text-to-video-synthesis',
            variant="fp16",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )        
        p = p.to(torch.device(device))        


        prompt = {"text": self.text_prompt}
        num_frames = {"num_frames": 10}
#        
        output_video_path = p(
            prompt=prompt,
            num_frames=nnum_frames,
            generator=generator,
        )[OutputKeys.OUTPUT_VIDEO]

        #p = pipeline('text-to-video-synthesis', model_dir)#, torch_dtype=torch.float16, variant="fp16")

#        test_text = {"text": self.text_prompt}
#        num_frames = {"num_frames": 10}
#        
#        output_video_path = p(
#            test_text,
#            num_frames,
#        )[OutputKeys.OUTPUT_VIDEO]
        # Run model

#        pipeline = DiffusionPipeline.pretrained_model_name_or_path(
#            model_dir,
#            variant="fp16",
#            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
#        )
#        pipeline = pipeline.to(torch.device(device))
#        if cpu_offload:
#            pipeline.enable_sequential_cpu_offload()
#        if attention_slice:
#        pipeline.enable_attention_slicing()

#        generator = torch.Generator(device=torch.device(device))#.manual_seed(seed)


#        output_video_path = pipeline(
#            prompt=prompt,
#            num_frames=num_frames,
#            #num_inference_steps=num_steps,
#            #height=height,
#            #width=width,
#            generator=generator,
#        )[OutputKeys.OUTPUT_VIDEO]



        filepath = bpy.path.abspath(output_video_path)
        if os.path.isfile(filepath):
            strip = scene.sequence_editor.sequences.new_movie(
                name=bpy.types.Scene.text_prompt,
                filepath=filepath,
                channel=1,
                frame_start=scene.frame_current,
            )
        else:
            print("Modelscope did not produce a file!")

        # Clean up memory
        torch.cuda.empty_cache()
            
        return {"FINISHED"}


class SequencerPanel(Panel):
    """Text to Video using ModelScope"""

    bl_idname = "SEQUENCER_PT_sequencer_panel"
    bl_label = "Text to Video"
    bl_space_type = "SEQUENCE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Text to Video"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        row = layout.row()
        row.prop(context.scene, "text_prompt", text="")
        row = layout.row()
        row.operator("sequencer.import_movie", text="Generate Movie")


def register():
    bpy.utils.register_class(SequencerImportMovieOperator)
    bpy.utils.register_class(SequencerPanel)
    bpy.types.Scene.text_prompt = bpy.props.StringProperty(
        name="text_prompt", default=""
    )


def unregister():
    bpy.utils.unregister_class(SequencerImportMovieOperator)
    bpy.utils.unregister_class(SequencerPanel)
    del bpy.types.Scene.text_prompt


if __name__ == "__main__":
    register()
