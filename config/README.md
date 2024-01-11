# args.json

You can edit the hyperparameters in the args.json in order to adjust the training process. All hyperparameters are explained in the following.

- **batch_size**: Training & Evaluation batch size.
- **inpainting_learning_rate**: Learning rate for pretraining. (Not used in our experiments)
- **learning_rate**: Learning rate for supervised training.
- **optimizer**: Optimizer. Valid options are "Adam" and "AdamW".
- **loss**: Loss function. Valid options are "Dice", "BCE", "Dice+BCE", "Tversky" and "FocalTversky".
- **num_frames**: Number of input frames to the model.
- **output_frames**: Output frames of the model. Follows python indexing conventions. Therefore, a negative number refers to the last n frames, while a positive number refers to one specific frame.
- **time_interval**: Number of space between consecutive frames. (1 = every frame, 2 = every second frame, etc.)
- **anchor_frame**: If true, the first frame in a sequence of num_frames is always the first frame from the whole clip (as used in PNSPlus).
- **normalize_mean**: Mean values for normalization of the input images.
- **normalize_std**: Standard deviation values for normalization of the input images.
- **img_height**: Image height.
- **img_width**: Image width.
- **random_crop_size**: Maximum number of pixels by which each side of the image is cropped.
- **random_flip_prob**: Probability that the image is flipped. Defines the probability for both, horizontal and vertical flips.
- **random_rotation_prob**: Probability that the image will be rotated.
- **random_roation_range**: Range of possible rotation angle.
- **random_blur_prob**: Probability that the image is blurred.
- **random_blur_radius**: Blurring radius.
- **random_brightness_probability**: Probability for randomly shifting the image brightness.
- **random_brightness_factor**: Minimum factor the brightness is multiplied with.
- **epochs**: Maximum number of training epochs.
- **pretrain_split**: Ratio between train and validation set for pretraining.
- **shuffle_data**: If true, the input data is shuffled.
- **patience**: Patience for early stopping. Defines after how many epochs without improvement on the validation loss the training is stopped.
- **amp**: If true, Pytorches automatic mixed precision mode is used for faster training. Does not work for models using the NS-Block.
- **n_workers**: Number of workers used for dataloading.
- **threshold**: Threshold for converting the model output into binary data.
- **save_predictions**: If true, the predicted segmentation maps are saved.
- **save_attention_maps**: If true, attention maps are generated and saved using medcam.
- **weight_decay**: Weight decay factor.
- **scheduler_gamma**: Gamma value for the exponential learning rate scheduler.
- **unique**: If true, each image is only used once as input to the model during testing. Set to false for models returning only one output per input image.
- **num_folds**: Number of folds for cross validation.
- **validation_fold**: Defines which fold should be used for validation.
- **loss_factors**: For models using deep supervision, the different losses can be weighted with different factors.
- **rect_dropout**: If true, random rectangles are cut out in the images in the pretraining dataset. The pretraining task is to reconstruct the dropped regions.
- **img_dropout**: Similar to rect_dropout, but instead of random rectangles, a complete image from each input sequence of num_frames is dropped.
- **use_wandb**: If true, the training progress and evaluation metrics are logged with WandB.