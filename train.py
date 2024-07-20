from diffusers.optimization import get_cosine_schedule_with_warmup
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F
import os
import shutil
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
from diffusers import UNet2DModel,DDPMScheduler, DDPMPipeline
from diffusers.utils import make_image_grid

kanji_file = "./makedata/hirakata.txt"
font_dir = "./fonts/ja"
image_size = (128, 128)
times = 20

for time in range(times):
    image_duplicate = time + 1
    image_duplicate = times + 1
    with open(kanji_file, "r", encoding="utf-8") as f:
        kanji_list = [kanji for line in f for kanji in line.strip()]  # 漢字をリスト化

    shutil.rmtree("./images", ignore_errors=True)  # 既存の画像データを削除

    for font_file in os.listdir(font_dir):
        print(f"Processing: {font_file}")
        if font_file.endswith(".ttf") or font_file.endswith(".TTF"):
            font_path = os.path.join(font_dir, font_file)
            font_name = os.path.splitext(font_file)[0]  # フォント名を取得

            # フォント名でフォルダを作成
            save_dir = os.path.join("images", font_name)  # imagesフォルダの下にフォント名フォルダ
            os.makedirs(save_dir, exist_ok=True)  # フォルダが存在しない場合は作成

            

            for kanji in kanji_list:
                for i in range(image_duplicate):
                    image = Image.new("RGB", image_size, color="white")
                    draw = ImageDraw.Draw(image)

                    font = ImageFont.truetype(font_path, size=110)  # フォントサイズ調整
                    # テキストの Bounding Box を取得
                    bbox = font.getbbox(kanji)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2 )

                    draw.text(position, kanji, font=font, fill="black")

                    image_name = f"{font_name}{str(i)}+{kanji}.png"
                    image_path = os.path.join(save_dir, image_name)  # 保存先をフォルダに変更
                    image.save(image_path)

    from dataclasses import dataclass

    @dataclass
    class TrainingConfig:
        image_size = 64
        train_batch_size = 16
        eval_batch_size = 16
        num_epochs = 50
        gradient_accumulation_steps = 1
        learning_rate = 1e-4
        lr_warmup_steps = 500
        save_image_epochs = 10
        save_model_epochs = 30
        mixed_precision = "fp16"
        output_dir = "output"

        push_to_hub = False
        overwrite_output_dir = True
        seed = 0

    config = TrainingConfig()

    from datasets import load_dataset

    config.dataset_name = "images/"
    config.output_dir = f"output/duplicates_{image_duplicate}"
    dataset = load_dataset(config.dataset_name, split="train")



    preprocess = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}


    dataset.set_transform(transform)



    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)



    model = UNet2DModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    sample_image = dataset[0]["images"].unsqueeze(0)
    print("Input shape:", sample_image.shape)

    print("Output shape:", model(sample_image, timestep=0).sample.shape)



    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    noise = torch.randn(sample_image.shape)
    timesteps = torch.LongTensor([50])
    noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)



    noise_pred = model(noisy_image, timesteps).sample
    loss = F.mse_loss(noise_pred, noise)



    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )



    def evaluate(config, epoch, pipeline):
        # Sample some images from random noise (this is the backward diffusion process).
        # The default pipeline output type is `List[PIL.Image]`
        images = pipeline(
            batch_size=config.eval_batch_size,
            generator=torch.Generator(device='cpu').manual_seed(config.seed), # Use a separate torch generator to avoid rewinding the random state of the main training loop
        ).images

        # Make a grid out of the images
        image_grid = make_image_grid(images, rows=4, cols=4)

        # Save the images
        test_dir = os.path.join(config.output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)
        image_grid.save(f"{test_dir}/{epoch:04d}.png")

    from accelerate import Accelerator
    from huggingface_hub import create_repo, upload_folder
    from tqdm.auto import tqdm
    from pathlib import Path
    import os

    def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
        # Initialize accelerator and tensorboard logging
        accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=os.path.join(config.output_dir, "logs"),
        )
        if accelerator.is_main_process:
            if config.output_dir is not None:
                os.makedirs(config.output_dir, exist_ok=True)
            if config.push_to_hub:
                repo_id = create_repo(
                    repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
                ).repo_id
            accelerator.init_trackers("train_example")

        # Prepare everything
        # There is no specific order to remember, you just need to unpack the
        # objects in the same order you gave them to the prepare method.
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )

        global_step = 0

        # Now you train the model
        for epoch in range(config.num_epochs):
            progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(train_dataloader):
                clean_images = batch["images"]
                # Sample noise to add to the images
                noise = torch.randn(clean_images.shape, device=clean_images.device)
                bs = clean_images.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                    dtype=torch.int64
                )

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

                with accelerator.accumulate(model):
                    # Predict the noise residual
                    noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                    loss = F.mse_loss(noise_pred, noise)
                    accelerator.backward(loss)

                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if accelerator.is_main_process:
                pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

                if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                    evaluate(config, epoch, pipeline)

                if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                    if config.push_to_hub:
                        upload_folder(
                            repo_id=repo_id,
                            folder_path=config.output_dir,
                            commit_message=f"Epoch {epoch}",
                            ignore_patterns=["step_*", "epoch_*"],
                        )
                    else:
                        pipeline.save_pretrained(config.output_dir)

    from accelerate import notebook_launcher

    args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

    notebook_launcher(train_loop, args, num_processes=1)

    import glob

    sample_images = sorted(glob.glob(f"{config.output_dir}/samples/*.png"))