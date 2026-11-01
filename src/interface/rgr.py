"""Assignment workspace reusing the project's local object-attribute CNN."""
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image
import streamlit as st

from src.rgr.inference import (ROOT, MODEL_PATH, Settings, SIZES, load_model,
                               transform_image, predict_image, evaluate_images,
                               metrics, parameter_sweeps, CLASSES)
from src.rgr.dataset import (demo_dataset, read_dataset_zip, split_samples,
                             dataset_fingerprint, evaluation_cohorts)
from .panels import group_board


@st.cache_resource(show_spinner="Loading the local shape CNN…")
def _model(path, revision):
    return load_model(path)


def _checkpoint(path):
    from src.rgr.provenance import file_sha256

    path = Path(path)
    return _model(str(path), file_sha256(path))


@st.cache_data(show_spinner=False)
def _demo():
    return demo_dataset(per_class=30, seed=2026)


def _image_id(image):
    array = np.asarray(image)
    return sha256(repr(array.shape).encode()+array.tobytes()).hexdigest()


def _cohorts(samples):
    return evaluation_cohorts(samples)


def _evaluate(samples, model, settings):
    rows = evaluate_images([s.image for s in samples], [s.label for s in samples], model, settings)
    return [{**row, "image_id": _image_id(sample.image), "image_name": sample.name}
            for row, sample in zip(rows, samples)]


def _comparison(samples, before, after, settings):
    initial = _evaluate(samples, before, settings)
    adapted = _evaluate(samples, after, settings) if after is not None else []
    return {"before": initial, "after": adapted,
            "metrics_before": metrics(initial), "metrics_after": metrics(adapted)}


def _table(rows, value=False):
    return [{**({"Setting": row["value"]} if value else
                 {"Image": row.get("image_name", row.get("index", "")), "Truth": row.get("truth", "")}),
             "Prediction": row["prediction"] or "No usable shape",
             "Confidence": round(row["confidence"], 4),
             "Decision": "Accepted" if row["accepted"] else "Rejected",
             **({"Correct": row["correct"]} if "correct" in row else {}),
             "Time (ms)": round(row["elapsed_ms"], 2)} for row in rows]


def _metric_table(comparison):
    rows = []
    for name, key in (("Pretrained CNN", "before"), ("Adapted CNN", "after")):
        values = comparison.get(key, [])
        if values:
            score = metrics(values)
            rows.append({"Model": name, "Images": score["total"], "Correct": score["correct"],
                         "Wrong accepted": score["errors"], "Rejected": score["rejected"],
                         "Accuracy": f"{score['accuracy']:.1%}", "Mean confidence": f"{score['mean_confidence']:.1%}"})
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption("Accuracy = correct accepted predictions / all images. Rejections count against accuracy and are shown separately from wrong accepted predictions. Softmax confidence is uncalibrated.")


def _available_training(directory, samples):
    from src.rgr.provenance import validate_training_record

    path = directory/"training.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        splits = validate_training_record(samples, report, directory/"model.keras")
        return report, splits
    except (OSError, ValueError):
        pass
    return None, None


def _single_photo_workspace():
    from src.rgr.photos import decode_photo, crop_photo
    from src.rgr.report import build_export, MODEL_COMPARISON
    from src.rgr.provenance import file_sha256, PREPROCESSING_ID

    st.write("Test one photograph immediately. A training ZIP is optional.")
    st.caption("Use one isolated shape on a light background. Crop near the outline with a small margin. This photo is kept separate from training and from the two five-image assignment cohorts.")
    source = st.radio("Photo input", ["Upload image", "Camera snapshot"], horizontal=True, key="rgr_photo_input")
    if source == "Upload image":
        uploaded = st.file_uploader("Photograph to test", type=["png", "jpg", "jpeg", "webp", "bmp"],
                                    max_upload_size=15, key="rgr_single_upload")
    else:
        uploaded = st.camera_input("Capture one shape", key="rgr_single_camera")
    if uploaded is None:
        st.info("Upload a photograph or capture one above to start testing.")
        return
    try:
        original = decode_photo(uploaded.getvalue())
        original_id = _image_id(original)
        with st.expander("Crop the photograph", expanded=False):
            st.caption("Crop bounds refer to the original, orientation-corrected photo. Brightness and contrast are applied afterwards.")
            horizontal = st.slider("Horizontal crop (%)", 0, 100, (0, 100), key="rgr_crop_x_"+original_id[:16])
            vertical = st.slider("Vertical crop (%)", 0, 100, (0, 100), key="rgr_crop_y_"+original_id[:16])
            st.image(original, caption="Original photograph", width=350)
        image, box = crop_photo(original, horizontal, vertical)
        truth_value = st.selectbox("Known shape / optional", ["Unknown", *CLASSES], key="rgr_truth_"+original_id[:16])
        truth = None if truth_value == "Unknown" else truth_value
    except (ValueError, OSError) as error:
        st.error(str(error))
        return
    try:
        before = _checkpoint(MODEL_PATH)
        directory = ROOT/"data"/"rgr"/"demo"
        demo_record, _ = _available_training(directory, _demo())
        after = _checkpoint(directory/"model.keras") if demo_record else None
    except (ImportError, ValueError, RuntimeError, OSError) as error:
        st.error(f"Could not load the local CNN: {error}")
        st.code("python -m pip install -r requirements-neural.txt", language="shell")
        return
    left, right = st.columns(2)
    with left:
        model_name = st.selectbox("Photo model", ["Pretrained CNN"]+(["Adapted CNN / synthetic demo"] if after is not None else []), key="rgr_photo_model")
        threshold = st.slider("Photo confidence threshold", .1, .9, .6, .05, key="rgr_photo_threshold")
        brightness = st.slider("Photo brightness (%)", -100, 100, 0, 5, key="rgr_photo_brightness")
        contrast = st.slider("Photo contrast", .5, 2., 1., .05, key="rgr_photo_contrast")
        size = st.select_slider("Photo processing size (pixels)", list(SIZES), value=224, key="rgr_photo_size")
        settings = Settings(threshold=threshold, brightness=brightness, contrast=contrast, size=size)
        model = before if model_name == "Pretrained CNN" else after
        result = predict_image(image, model, settings)
        st.markdown(f"**Prediction: {result['prediction'] or 'No usable shape'} · {result['confidence']:.1%}**")
        if result["accepted"]:
            st.success(f"Accepted at threshold {threshold:.2f}.")
        else:
            st.warning("Rejected: "+result.get("reason", "unresolved").replace("_", " "))
        if truth:
            st.caption("Correct for your label." if result["accepted"] and result["prediction"] == truth else
                       "Not correct for your label: the prediction is wrong or was rejected.")
        else:
            st.caption("No ground truth supplied; correctness has not been assessed.")
        st.caption(f"Preprocessing + inference: {result['elapsed_ms']:.2f} ms. Softmax confidence is uncalibrated.")
    with right:
        st.image(transform_image(image, settings), caption=f"Actual processed crop · {size} × {size}", width=min(440, size*2))
        st.caption("The CNN consumes a 48 × 48 grayscale tensor after this processing. The bundled adapted model was trained on synthetic drawings.")
        if result["probabilities"]:
            st.bar_chart([{"Class": name, "Probability": value} for name, value in result["probabilities"].items()],
                         x="Class", y="Probability", height=160, color="#002FA7")
    identity = sha256(repr((original_id, box, truth, model_name, asdict(settings),
                           file_sha256(MODEL_PATH), demo_record.get("adapted_model_sha256") if demo_record else None)).encode()).hexdigest()
    if st.button("Measure photo parameters", type="primary", key="rgr_measure_photo"):
        with st.spinner("Testing the photograph under each parameter setting…"):
            sweeps = parameter_sweeps(image, model, settings)
        if truth:
            for rows in sweeps.values():
                for row in rows:
                    row.update(truth=truth, correct=row["accepted"] and row["prediction"] == truth)
        st.session_state.rgr_photo_measurements = {"identity": identity, "sweeps": sweeps}
    saved = st.session_state.get("rgr_photo_measurements", {})
    valid = saved.get("identity") == identity
    if valid:
        for key, title in (("threshold", "Confidence"), ("brightness", "Brightness"), ("contrast", "Contrast"),
                           ("size", "Size and time"), ("augmentation", "Augmentation")):
            with st.expander(title+" / photo results", expanded=key == "threshold"):
                st.dataframe(_table(saved["sweeps"][key], value=True), hide_index=True, width="stretch")
                if key == "augmentation":
                    st.image(group_board([transform_image(image, options) for options in
                        (settings, replace(settings, rotation=20), replace(settings, flip=True))],
                        ["Original", "Rotation +20°", "Horizontal flip"], tile=160), width="stretch")
    elif saved:
        st.caption("The photo, crop, label, model or controls changed. Measure photo parameters again to update the tables.")
    record = {"image_id": _image_id(image), "image_name": uploaded.name, **result}
    if truth:
        record.update(truth=truth, correct=result["accepted"] and result["prediction"] == truth)
    payload = {"metadata": {"source": source, "dataset_kind": "single_photo", "settings": asdict(settings),
               "test_image": uploaded.name, "test_image_id": _image_id(image), "original_image_id": original_id,
               "crop_box_pixels": box, "original_size": [original.shape[1], original.shape[0]],
               "test_image_kind": "personal_photo", "known_shape": truth, "model": model_name,
               "model_sha256": file_sha256(MODEL_PATH) if model_name == "Pretrained CNN" else demo_record["adapted_model_sha256"],
               "preprocessing_id": PREPROCESSING_ID,
               "sliders": ["confidence", "brightness", "contrast"],
               "note": "One photo for parameter exploration. It is not a five-image evaluation or a personal training dataset."},
               "single_image": [record], "model_comparison": MODEL_COMPARISON,
               "training": demo_record if model_name != "Pretrained CNN" else None,
               "sweeps": saved["sweeps"] if valid else {}}
    illustrations = {"original.png": original, "crop.png": image, "processed.png": transform_image(image, settings)}
    if valid:
        for name, options in (("original", settings), ("rotate20", replace(settings, rotation=20)),
                              ("hflip", replace(settings, flip=True))):
            illustrations[f"augmentation/{name}.png"] = transform_image(image, options)
    st.download_button("Download photo experiment", build_export(payload, illustrations),
                       "logicapture_photo_experiment.zip", "application/zip", key="rgr_photo_export")


def rgr_workspace():
    from src.rgr.report import build_export, MODEL_COMPARISON

    st.subheader("РГР / Model testing")
    st.write("Explore how image settings affect recognition, adapt the existing shape CNN, and compare the same unseen images before and after training.")
    st.caption("One tightly cropped shape on a light background: circle, triangle, quadrilateral or polygon. This is the perception experiment for Logicapture.")

    source = st.radio("Experiment dataset", ["Synthetic demo", "Your labeled photographs", "Single photograph"],
                      horizontal=True, key="rgr_source")
    if source == "Single photograph":
        _single_photo_workspace()
        return
    if source == "Synthetic demo":
        samples = _demo()
        kind = "synthetic"
        st.info("The demo uses generated drawings. It exercises all controls and the training workflow; the assignment's personal-photograph tests require your own images.")
    else:
        st.caption("Upload a ZIP with class folders: circle/, triangle/, quadrilateral/, polygon/. Include at least 5 distinct images per class; about 15 per class leaves enough held-out photos for both five-image experiments. Use one photo per independent drawing; exclude alternate shots or edited copies before importing.")
        uploaded = st.file_uploader("Labeled photograph dataset (.zip)", type=["zip"], max_upload_size=32, key="rgr_dataset")
        if not uploaded:
            st.code("dataset/\n  circle/photo01.jpg\n  triangle/photo01.jpg\n  quadrilateral/photo01.jpg\n  polygon/photo01.jpg", language=None)
            return
        try:
            samples = read_dataset_zip(uploaded.getvalue())
        except (ValueError, OSError) as error:
            st.error(str(error))
            return
        kind = "personal_photos"
    try:
        splits = split_samples(samples, seed=2026)
        fingerprint = dataset_fingerprint(samples)
    except ValueError as error:
        st.error(str(error))
        return
    run_root = ROOT/"outputs"/"rgr"/fingerprint/"runs"
    directories = sorted(run_root.glob("*"), key=lambda path: path.stat().st_mtime_ns, reverse=True) if run_root.is_dir() else []
    if kind == "synthetic":
        directories.append(ROOT/"data"/"rgr"/"demo")
    directory, training = ROOT/"data"/"rgr"/"demo", None
    invalid_run = False
    for candidate in directories:
        record, recorded_splits = _available_training(candidate, samples)
        if record:
            directory, training, splits = candidate, record, recorded_splits
            break
        invalid_run |= (candidate/"training.json").is_file()
    initial, final = _cohorts(splits["test"])
    adapted_path = directory/"model.keras"
    counts = {name: len(items) for name, items in splits.items()}
    st.caption(f"{len(samples)} images · train {counts['train']} / validation {counts['validation']} / test {counts['test']} · test images never enter training or epoch selection")
    if training:
        st.caption(f"Saved split seed: {training['seed']} · source identities and both checkpoint hashes verified")
    elif invalid_run:
        st.info("An earlier saved run could not be verified against these images, the current model files and the recorded split. Train again to create a verified run.")
    if len(initial) < 5 or len(final) < 5:
        st.warning("The test split needs at least 10 images for two separate five-image cohorts. Add more source photographs to complete both assignment comparisons.")
    if not MODEL_PATH.is_file():
        st.error("The existing pretrained checkpoint is missing: data/models/attributes.keras.")
        st.code("python -m src.vision.attribute_cnn --epochs 12 --per-class 90", language="shell")
        return
    try:
        before = _checkpoint(MODEL_PATH)
        after = _checkpoint(adapted_path) if training else None
    except (ImportError, ValueError, RuntimeError, OSError) as error:
        st.error(f"Could not load the local CNN: {error}")
        st.code("python -m pip install -r requirements-neural.txt", language="shell")
        return
    from src.rgr.provenance import file_sha256, PREPROCESSING_ID
    model_revision = (file_sha256(MODEL_PATH), training["adapted_model_sha256"] if training else None)

    testing, fitting, reporting = st.tabs(["Test image & sliders", "Train & compare", "Assignment & export"])
    with testing:
        choices = {sample.name: sample for sample in splits["test"]}
        default_index = next((index for index, item in enumerate(choices.values())
                              if kind == "synthetic" and item.label == "quadrilateral"), 0)
        selected = st.selectbox("Held-out test image", list(choices), index=default_index, key="rgr_test_image")
        sample = choices[selected]
        columns = st.columns(2)
        with columns[0]:
            model_name = st.selectbox("Model", ["Pretrained CNN"]+(["Adapted CNN"] if after is not None else []), key="rgr_model")
            threshold = st.slider("Confidence threshold", .1, .9, .6, .05, key="rgr_threshold")
            brightness = st.slider("Brightness (%)", -100, 100, 0, 5, key="rgr_brightness")
            contrast = st.slider("Contrast", .5, 2., 1., .05, key="rgr_contrast")
            size = st.select_slider("Processing size (pixels)", options=list(SIZES), value=224, key="rgr_size")
            settings = Settings(threshold=threshold, brightness=brightness, contrast=contrast, size=size)
            current_model = before if model_name == "Pretrained CNN" else after
            result = predict_image(sample.image, current_model, settings)
            st.markdown(f"**Prediction: {result['prediction'] or 'No usable shape'} · {result['confidence']:.1%}**")
            if result["accepted"]:
                st.success(f"Accepted at threshold {threshold:.2f}. Ground truth: {sample.label}.")
            else:
                st.warning(f"Rejected: {result.get('reason', 'unresolved').replace('_', ' ')}. Ground truth: {sample.label}.")
            st.caption(f"Preprocessing + inference: {result['elapsed_ms']:.2f} ms. Model loading excluded.")
        with columns[1]:
            st.image(transform_image(sample.image, settings), caption=f"Actual processed image · {size} × {size}", width=min(440, size*2))
            st.caption("The CNN then receives a 48 × 48 grayscale tensor. The size experiment changes intermediate resampling and processing cost; it does not change the CNN architecture. Brightness is an additive percentage of full pixel range.")
            if result["probabilities"]:
                st.bar_chart([{"Class": key, "Probability": value} for key, value in result["probabilities"].items()],
                             x="Class", y="Probability", height=180, color="#002FA7")
            else:
                st.caption("No class probabilities for a blank or saturated image.")
        identity = sha256(repr((fingerprint, selected, model_name, asdict(settings), model_revision)).encode()).hexdigest()
        if st.button("Run parameter experiments", key="rgr_run_sweeps", type="primary"):
            with st.spinner("Measuring threshold, lighting, resolution and augmentation…"):
                sweeps = parameter_sweeps(sample.image, current_model, settings)
            for rows in sweeps.values():
                for row in rows:
                    row["truth"] = sample.label
                    row["correct"] = row["accepted"] and row["prediction"] == sample.label
            st.session_state.rgr_sweeps = {"identity": identity, "data": sweeps, "image_id": _image_id(sample.image), "model": model_name}
        saved = st.session_state.get("rgr_sweeps", {})
        if saved.get("identity") == identity:
            st.caption("Each experiment varies one setting and holds the other current controls fixed. Inference timings are single-run measurements and can vary.")
            for key, title in (("threshold", "Confidence threshold"), ("brightness", "Brightness"),
                               ("contrast", "Contrast"), ("size", "Image size"), ("augmentation", "Data augmentation")):
                with st.expander(title+" / measured results", expanded=key == "threshold"):
                    st.dataframe(_table(saved["data"][key], value=True), hide_index=True, width="stretch")
                    if key == "augmentation":
                        variants = [replace(settings, rotation=0, flip=False), replace(settings, rotation=20, flip=False), replace(settings, rotation=0, flip=True)]
                        st.image(group_board([transform_image(sample.image, variant) for variant in variants], ["Original", "Rotation +20°", "Horizontal flip"], tile=160), width="stretch")
        elif saved:
            st.caption("The image, model or controls changed. Run parameter experiments again to update the measured tables.")

    with fitting:
        st.write("Adapt a fresh copy of the pretrained CNN. Training uses only the training split; validation selects the saved epoch. Both comparisons use the same held-out images.")
        epochs = st.slider("Training epochs", 1, 20, 4, key="rgr_epochs")
        if st.button("Train on this dataset", key="rgr_train", type="primary"):
            from src.rgr.training import fine_tune
            status = st.empty()
            try:
                fine_tune(samples, run_root/uuid4().hex, epochs=epochs, seed=2026, progress=status.caption)
                st.rerun()
            except (ValueError, RuntimeError, ImportError, OSError) as error:
                st.error(f"Training failed: {error}")
            finally:
                status.empty()
        if training:
            st.success(f"Saved adapted model · epoch {training.get('selected_epoch', '?')} selected by validation")
            st.caption("The training record reports unthresholded argmax accuracy. The comparisons below use the current confidence threshold and count rejected images against accuracy.")
            if training.get("history"):
                st.line_chart(training["history"], x="epoch", y=["train_loss", "validation_loss"],
                              color=["#002FA7", "#777777"])
            st.download_button("Download training record", json.dumps(training, indent=2, ensure_ascii=False), "rgr_training.json", "application/json", key="rgr_training_export")
        else:
            st.info("Train once to enable the adapted model and Before → After comparisons.")
        evaluation_identity = (fingerprint, training["seed"] if training else 2026, model_revision, asdict(settings))
        if st.button("Compare held-out images", key="rgr_compare"):
            with st.spinner("Evaluating both models on the same held-out cohorts…"):
                st.session_state.rgr_comparison = {"identity": evaluation_identity,
                    "initial": _comparison(initial, before, after, settings),
                    "final": _comparison(final, before, after, settings)}
        comparison = st.session_state.get("rgr_comparison", {})
        if comparison.get("identity") == evaluation_identity:
            for cohort, title, cohort_samples in (("initial", f"Initial test images / {len(initial)} of 5", initial),
                                                  ("final", f"New final test images / {len(final)} of 5", final)):
                st.markdown(f"**{title}**")
                checked = comparison[cohort]
                _metric_table(checked)
                if cohort_samples:
                    st.image(group_board([transform_image(s.image, settings) for s in cohort_samples], [s.label for s in cohort_samples], tile=145), width="stretch")
                with st.expander("Per-image measurements / "+cohort):
                    st.caption("Pretrained model")
                    st.dataframe(_table(checked["before"]), hide_index=True, width="stretch")
                    if checked["after"]:
                        st.caption("Adapted model")
                        st.dataframe(_table(checked["after"]), hide_index=True, width="stretch")
                if cohort == "initial" and checked["after"]:
                    for sample_index, (old, new) in enumerate(zip(checked["before"][:3], checked["after"][:3])):
                        left, right = st.columns(2)
                        picture = transform_image(cohort_samples[sample_index].image, settings)
                        left.image(picture, caption=f"Before · {old['prediction']} · {old['confidence']:.1%}", width=200)
                        right.image(picture, caption=f"After · {new['prediction']} · {new['confidence']:.1%}", width=200)
        elif comparison:
            st.caption("The dataset, trained model or controls changed. Compare again to update the tables.")

    with reporting:
        st.markdown("**Model selection / task 1**")
        st.dataframe(MODEL_COMPARISON, hide_index=True, width="stretch")
        st.write("The compact attribute CNN is selected because its existing four-class shape head and local checkpoint already fit the perception task. Adaptation keeps the original checkpoint intact. A larger natural-image classifier needs a new task-specific head and a separate evaluation.")
        st.markdown("[MobileNetV2 documentation](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2) · [TensorFlow transfer learning](https://www.tensorflow.org/tutorials/images/transfer_learning)")
        st.markdown("**Assignment evidence**")
        comparison = st.session_state.get("rgr_comparison", {})
        valid_comparison = comparison.get("identity") == evaluation_identity
        saved = st.session_state.get("rgr_sweeps", {})
        valid_sweeps = saved.get("identity") == identity
        st.dataframe([
            {"Tasks": "1", "Evidence": "Model comparison and selection", "Status": "Available"},
            {"Tasks": "2, 9, 10", "Evidence": "Two disjoint five-image test cohorts and Before → After", "Status":
                ("Measured" if len(initial) == len(final) == 5 else f"Partial / {len(initial)} initial and {len(final)} final images")
                if valid_comparison and after is not None else "Run training and Compare held-out images"},
            {"Tasks": "3–7", "Evidence": "Three sliders, size, parameter tables and two augmentations", "Status": "Measured" if valid_sweeps else "Run parameter experiments"},
            {"Tasks": "8", "Evidence": "Stratified split, saved model and validation history", "Status": "Saved" if training else "Train on this dataset"},
            {"Tasks": "Photographs", "Evidence": "Personal photos required by the supplied assignment", "Status": "User-supplied photos" if kind == "personal_photos" else "Pending / synthetic demonstration"}
        ], hide_index=True, width="stretch")
        payload = {"metadata": {"source": source, "dataset_kind": kind, "settings": asdict(settings),
                                  "dataset_fingerprint": fingerprint, "split_counts": counts,
                                  "split_seed": training["seed"] if training else 2026,
                                  "base_model_sha256": model_revision[0], "adapted_model_sha256": model_revision[1],
                                  "preprocessing_id": PREPROCESSING_ID,
                                  "test_image": sample.name, "test_image_id": _image_id(sample.image),
                                  "sweep_model": saved.get("model") if valid_sweeps else None,
                                  "cnn_input": [48, 48], "sliders": ["confidence", "brightness", "contrast"],
                                  "before_after_visual_count": min(3, len(initial)) if after is not None and valid_comparison else 0},
                   "model_comparison": MODEL_COMPARISON,
                   "sweeps": saved["data"] if valid_sweeps else {},
                   "training": training,
                   "initial": comparison["initial"] if valid_comparison else {},
                   "final": comparison["final"] if valid_comparison else {}}
        illustrations = {"selected_original.png": sample.image,
                         "selected_processed.png": transform_image(sample.image, settings)}
        if valid_sweeps:
            for name, options in (("original", replace(settings, rotation=0, flip=False)),
                                  ("rotate20", replace(settings, rotation=20, flip=False)),
                                  ("hflip", replace(settings, rotation=0, flip=True))):
                illustrations[f"augmentation/{name}.png"] = transform_image(sample.image, options)
        if valid_comparison:
            for index, item in enumerate(initial[:3], start=1):
                picture = transform_image(item.image, settings)
                row = comparison["initial"]["before"][index-1]
                illustrations[f"before_after/{index}_before.png"] = group_board([picture],
                    [f"Before: {row['prediction']} / {row['confidence']:.1%}"], columns=1, tile=280)
                if after is not None:
                    row = comparison["initial"]["after"][index-1]
                    illustrations[f"before_after/{index}_after.png"] = group_board([picture],
                        [f"After: {row['prediction']} / {row['confidence']:.1%}"], columns=1, tile=280)
        st.download_button("Download РГР report & measurements", build_export(payload, illustrations),
                           "logicapture_rgr.zip", "application/zip", key="rgr_report_export")
        st.caption("The ZIP contains the report, CSV measurements, JSON provenance and comparison images. Missing experiments remain marked incomplete. Exact duplicate pixels cannot cross dataset splits; exclude alternate shots and edited copies of the same drawing before importing.")
