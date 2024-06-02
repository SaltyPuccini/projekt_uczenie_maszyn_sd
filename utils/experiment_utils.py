import pathlib

SAVE_LOCATION_PREFIX = pathlib.Path("/media/kwoj/borrowed/Projekt_Uczenie_Maszyn")

GUIDANCE_SCALE = 0.0
WARM_UP_ITERATIONS = 10
WARM_UP_STEPS = 5
EXPERIMENT_ITERATIONS = 10

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
    "space-station": "space station on the moon.",
}

PROMPT_PREFIX = ("Generate a top-down view of a location in a Japanese Role-Playing Game (J-RPG). The scene should be "
                 "in a pixel art style and follow a tile-based design. The view must be directly from above. The "
                 "location is a ")

WARM_UP_PROMPT = PROMPT_PREFIX + prompts_dict["forest-village"]

inference_steps = [1, 5, 10, 20, 50, 100]
