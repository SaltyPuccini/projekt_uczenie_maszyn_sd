import pathlib
SAVE_LOCATION_PREFIX = pathlib.Path("/media/kwoj/borrowed/Projekt_Uczenie_Maszyn")

WARM_UP_ITERATIONS = 2
EXPERIMENT_ITERATIONS = 10
PROMPT_PREFIX = ("Generate a top-down view of a location in a Japanese Role-Playing Game (J-RPG). The scene should be "
                 "in a pixel art style and follow a tile-based design. The view must be directly from above. The "
                 "location is a")
models_dict = {
    "sdxl-turbo": "stabilityai/sdxl-turbo",
    "sd-2-1": "stabilityai/stable-diffusion-2-1",
}
prompts_dict = {
    "forest-village": "quaint, bustling village nestled in a lush forest.",
    "desert-town": "desert oasis surrounded by palm trees.",
    "mountain-lake": "mountain path overseeing a serene lake.",
    "coastal-town": "deserted island with a treasure chest and wrecked ship.",
    "medieval-city": "medieval city.",
    "path-to-castle": "path through a dark forest leading to a castle gates.",
}
inference_steps = [1, 5, 10, 20, 50, 100]
guidance_scales = [0.0, 3.0, 7.0, 10.0, 13.0, 17.0, 20.0]