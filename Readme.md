# Image Generation Tool using 🎨 Diffusion Models

Welcome to the Image Generation Tool, leveraging the power of diffusion models to create high-quality, realistic images. Diffusion models have emerged as a groundbreaking approach in the field of generative models, often surpassing the performance of traditional Generative Adversarial Networks (GANs).

## Table of Contents
- [Introduction](#introduction)
- [How Diffusion Models Work](#how-diffusion-models-work)
- [Using Hugging Face for Diffusion Models](#using-hugging-face-for-diffusion-models)
- [Generating Images with Dream-like Diffusion](#generating-images-with-dream-like-diffusion)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Diffusion models generate images through a process of iterative noise addition and removal. By training on this process, these models learn to produce highly realistic images. Our tool utilizes pre-trained diffusion models from Hugging Face, specifically the Dream-like Diffusion 1.0 model, to simplify and enhance the image generation experience.

## How Diffusion Models Work

Diffusion models operate by:
1. **Adding Noise:** Starting with a clear image, noise is gradually added to it.
2. **Training to Reverse Noise:** The model learns to reverse the process, predicting the original clear image from the noisy one.
3. **Iterative Process:** This iterative process of adding and removing noise enables the generation of new, high-quality images.

## Using Hugging Face for Diffusion Models

Hugging Face is a leading machine-learning community that offers a wide range of pre-trained models, including diffusion models. The Hugging Face Diffusers library provides an easy-to-use interface for these models, allowing for seamless integration and image generation.

## Generating Images with Dream-like Diffusion

The Dream-like Diffusion 1.0 model from Hugging Face enables the generation of realistic images based on text prompts. Key parameters that can be adjusted include:
- **Number of Inference Steps:** Higher steps improve quality but increase computation time.
- **Negative Prompting:** Helps refine the output by guiding the model on what not to include.
- **Image Dimensions:** Customize the height and width of the generated images.
- **Batch Generation:** Specify the number of images to generate per prompt.

## Features

- **High-Quality Image Generation:** Leveraging the strengths of diffusion models for superior image quality.
- **Customizable Parameters:** Fine-tune the image generation process with adjustable parameters.
- **User-Friendly Interface:** Intuitive and easy-to-use, even for those new to diffusion models.
- **Pre-Trained Models:** Utilize robust, pre-trained models from Hugging Face for efficient image generation.

## Tech Stack

- **Languages:** Python
- **Libraries:** 
  - Hugging Face Diffusers
  - Transformers
  - PyTorch
- **APIs:** Hugging Face Hub

## Usage

To generate images using the Dream-like Diffusion model:

1. **Install the required libraries:**
    ```bash
    pip install diffusers transformers torch
    ```

2. **Load the pre-trained model:**
    ```python
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained("huggingface/dreamlike-diffusion-1.0")
    ```

3. **Generate an image from a text prompt:**
    ```python
    prompt = "A serene landscape with mountains and a river"
    images = pipe(prompt, num_inference_steps=50, height=512, width=512, num_images_per_prompt=1)
    ```

4. **Save or display the generated image:**
    ```python
    images[0].save("generated_image.png")
    ```

## Examples

Here are a few example prompts and their generated images:

- **Prompt:** "A futuristic city skyline at sunset"
  ![image1](https://github.com/RSN601KRI/ImageGenerationTool/assets/106860359/6efb0ffe-9fca-42c1-b670-671c969684c6)


- **Prompt:** "A vibrant forest in autumn"
  ![image2](https://github.com/RSN601KRI/ImageGenerationTool/assets/106860359/80905ff8-d692-4fec-83e4-17b73044d327)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

