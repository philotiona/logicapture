# РГР №1 model testing

The РГР workspace implements the supplied **Есептік-графикалық жұмыс №1 — Объектілерді тану моделін таңдау, бейімдеу және оқыту** assignment inside Logicapture. Start the existing application and select **РГР coursework** in the navigation bar:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

It reuses `src/vision/attribute_cnn.py` and the project's existing `data/models/attributes.keras` checkpoint. No Gradio or scikit-learn installation is needed. TensorFlow is loaded only when entering the РГР workspace. Install `requirements-neural.txt` if it is missing.

The model recognizes one tightly cropped circle, triangle, quadrilateral, or polygon on a light background. The polygon training class is based on pentagons; it does not imply recognition of arbitrary visual concepts. Coursework is separate from the main Bongard and image-sorting workflows.

## Assignment workflow

1. In **Assignment & export**, review the comparison of the existing compact CNN, MobileNetV2, and the OpenCV alternative. The compact CNN is selected because its existing trained shape head matches this project's perception task. MobileNetV2 is considered qualitatively; no performance result is claimed for it.
2. Start with **Synthetic demo** to exercise the interface, or upload **Your labeled photographs**. The demo contains generated sketches and is explicitly not a real-photograph dataset.
3. In **Test image & sliders**, select an unseen image and a model. Change confidence, brightness, contrast, and processing size. The displayed processed image is the one used by the inference pipeline. Thresholding changes whether a prediction is accepted; it does not change the network's underlying class probabilities.
4. Press **Run parameter experiments** to record all required confidence, brightness, contrast and size values, including elapsed time. Original, 20-degree rotation, and horizontal flip provide the augmentation comparison. Each sweep varies one control while retaining the others at their displayed settings. Changing any control invalidates the displayed/exported sweep.
5. In **Train & compare**, train a new copy of the existing CNN. The original checkpoint is preserved. Training changes the shape head, with training-only augmentation; validation chooses the saved epoch. Test images never enter training or checkpoint selection.
6. Press **Compare held-out images**. The initial five images are shared between pretrained and adapted models; the final five are a separate cohort. At least three Before → After visual comparisons accompany the table. Correct, wrong accepted, rejected, mean confidence, and accuracy are reported. Rejections count against accuracy.
7. Download **РГР report & measurements**. The ZIP contains `REPORT.md`, CSV measurements, JSON provenance, and annotated comparison images. Missing measurements and the personal-photo requirement are marked incomplete rather than inferred from synthetic results.

## Personal dataset

### Test one photograph first

Select **Single photograph** in the РГР workspace to upload one image or use a camera snapshot. No training ZIP is needed. Open **Crop the photograph** to trim a phone photo near its shape; the displayed crop is the input used by the sliders and classifier. Crop bounds and original image identity are recorded in the photo report.

The true shape is optional. With **Unknown** selected, the report records predictions and confidence without claiming correctness. **Measure photo parameters** runs the required sweeps on that photograph, and **Download photo experiment** exports its readings and images. Changing the image, crop, label, model or controls invalidates the prior sweep. Removing or replacing an invalid input does not leave its old prediction visible.

This photograph is separate from the training dataset and from the two five-image evaluation cohorts. The bundled adapted model remains a synthetic-demo model. Use **Your labeled photographs** and the ZIP workflow below for personal fine-tuning and the complete assignment comparison.

### Prepare a training dataset

Upload a ZIP containing labeled images, with or without the outer `dataset/` folder:

```text
dataset/
  circle/photo01.jpg
  triangle/photo01.jpg
  quadrilateral/photo01.jpg
  polygon/photo01.jpg
```

Use at least two represented classes and five unique source images per represented class. For both five-image test cohorts, use approximately 15 photographs per class across four classes, or more if fewer classes are represented. Include more varied training photographs when practical. Photograph isolated shapes on light paper and crop near the shape with a small margin, matching the object-region model. Folder names supply ground truth and never enter the model input.

Splitting is stratified, approximately 60% training, 20% validation, and 20% test. Exact duplicate decoded pixels remain in a single split and are deduplicated when forming the two evaluation cohorts. Edited versions, multiple shots of the same drawing, or other near duplicates cannot be identified reliably by a pixel hash: prepare independent sources before importing. Uploaded images are decoded in memory and are not extracted to arbitrary paths. Personal training outputs are saved under ignored `outputs/rgr/` directories; every run gets a new directory.

The supplied assignment requires real new photographs. No personal photographs were provided here, so the bundled runnable experiment uses synthetic drawings. Upload your photos and rerun training, both comparisons, and parameter experiments before treating the report as a completed personal-photo submission.

## Interpreting the controls

- **Brightness** adds −100 to +100 percent of the full 255-level pixel range, with clipping. The endpoints can erase the shape and are rejected as unusable images.
- **Contrast** scales pixel differences around mid-gray, from 0.5 to 2.0. There is no automatic contrast correction after applying the slider.
- **Confidence** is a class-softmax acceptance threshold. High confidence does not establish that an out-of-distribution image is correctly recognized.
- **Processing size** chooses 128, 224, 320, 416 or 640 square pixels with aspect-preserving padding. The existing CNN still consumes a 48×48 grayscale tensor. This experiment measures resampling effects and end-to-end preprocessing/inference cost, not a change to the network's resolution. Model loading is excluded from timings; a single timing can vary due to runtime warmup and system load.

The pretrained checkpoint was trained on independent synthetic isolated objects. Adapting only its shape head is a deliberately small local experiment; improvement is measured rather than guaranteed. Neither the demo nor this assignment establishes general Bongard reasoning performance.

## Command-line reproduction

The bundled synthetic training run uses 30 generated sources per class, seed 2026, and four epochs:

```powershell
.\.venv\Scripts\python.exe -m src.rgr.training --demo --per-class 30 --seed 2026 --epochs 4 --output outputs/rgr-reproduction
```

To adapt to your own dataset:

```powershell
.\.venv\Scripts\python.exe -m src.rgr.training --dataset my_photos.zip --epochs 8 --output outputs/rgr-personal
```

Choose an unused output directory. Each saved `training.json` records the source identity of every split, base-model hash, training and validation losses, selected epoch, and identical-test-set before/after results. CLI outputs can be inspected independently; the app discovers its own saved runs by dataset fingerprint.

Saved runs also record the adapted checkpoint hash and preprocessing identity. The app and report CLI verify both checkpoint files and the complete source split manifest before using an adapted run. Evaluation follows the recorded training seed, including custom seeds, so changing a seed cannot silently turn training sources into test sources. Legacy records without these checks must be retrained; the bundled demo record has been verified against its actual weights and readings.

To reproduce the complete bundled measurement report with its two five-image cohorts, tables, augmentation images, and three Before → After pairs:

```powershell
.\.venv\Scripts\python.exe -m src.rgr.experiment --demo --output outputs/rgr-demo-report
```

The included demonstration report is in `data/rgr/demo/experiment.zip`. Training-record accuracy uses unthresholded argmax; interactive report accuracy applies the selected confidence threshold and counts rejections against accuracy.

## Assignment coverage

| Task | Implementation | Evidence |
|---|---|---|
| 1 | Model comparison and justification | Assignment tab and report |
| 2 | Pretrained model on first five held-out images | Per-image truth, class, confidence and correctness |
| 3 | Confidence threshold 0.1, 0.3, 0.5, 0.7, 0.9 | Live slider and measured sweep |
| 4 | Brightness −100, −50, 0, +50, +100 | Live slider, processed pixels and measured sweep |
| 5 | Contrast 0.5, 1, 2 | Live slider and measured sweep |
| 6 | 128, 224, 320, 416, 640 | Size selector, timing and confidence |
| 7 | Rotation and horizontal flip | Original plus two transformed images and measurements |
| 8 | Personal dataset adaptation | Separate train/validation/test, saved model and history |
| 9 | Same initial five before/after | Paired measurements and three visual comparisons |
| 10 | Five new held-out sources and three sliders | Separate final cohort and final metrics |

The photograph-dependent rows require the user's personal dataset; synthetic demonstration status is retained in exported evidence.

Primary model references: [Keras MobileNetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2), [TensorFlow transfer learning](https://www.tensorflow.org/tutorials/images/transfer_learning).
