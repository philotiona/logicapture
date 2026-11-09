"""Teach, reuse, review and export a local two-group image task."""
from hashlib import sha256
import json
from uuid import uuid4

import streamlit as st

from src.image_tasks import ImageTask, export_results, fit_task, load_task, read_example_sheet, read_images, sort_batch
from src.vision.pretrained import model_directory
from .panels import group_board


def _files(uploads):
    return [(file.name, file.getvalue()) for file in uploads or []]


def _identity(files):
    digest = sha256()
    for name, data in files:
        digest.update(json.dumps([name, len(data)], ensure_ascii=False).encode())
        digest.update(data)
    return digest.hexdigest()


def _read_cached(files, key):
    identity = _identity(files)
    cached = st.session_state.get(key)
    if cached is None or cached[0] != identity:
        cached = identity, read_images(files)
        st.session_state[key] = cached
    items, skipped = cached[1]
    if skipped:
        st.caption(f"Skipped {len(skipped)} files that are not supported images.")
    return items


def _task_settings(*, sheet=False):
    suffix = "_sheet" if sheet else ""
    name = st.text_input("Name this setup", "Example sheet" if sheet else "My image check", max_chars=120,
                         key="image_task_name"+suffix)
    left, right = st.columns(2)
    a = left.text_input("First group name", "Group A" if sheet else "Keep", max_chars=60, key="image_task_a_name"+suffix)
    b = right.text_input("Second group name", "Group B" if sheet else "Other", max_chars=60, key="image_task_b_name"+suffix)
    available = model_directory("siglip") is not None
    representation = st.radio("Image type", ["Photographs and screenshots", "Black-and-white drawings"],
                              index=0 if sheet or available else 1, horizontal=True, key="image_task_type"+suffix)
    photos = representation == "Photographs and screenshots"
    relation = False
    if photos:
        methods = ["Visual similarity", "Action or relationship"]
        method = st.radio("What separates the groups?", methods, index=int(sheet), horizontal=True,
                          key="image_task_photo_method"+suffix)
        relation = method == methods[1]
        if relation:
            st.caption("Find a visible condition, then check it against every reference before sorting. This takes longer than visual similarity.")
        model = "qwen" if relation else "siglip"
        if model_directory(model) is None:
            st.info(f"This method needs the optional local {'Qwen vision' if relation else 'SigLIP'} model.")
            st.code(f"python -m pip install -r requirements-general.txt\npython -m src.vision.pretrained --download {model}",
                    language="shell")
    return {"name": name.strip(), "groups": (a.strip(), b.strip()),
            "representation": "photos" if photos else "drawings", "photo_relation": relation}


def _new_task():
    settings = _task_settings()
    st.caption("Upload 4–6 varied examples per group (2–12 supported). Include the differences you want ignored, such as angle or background. Names are labels for you; the model receives pixels only.")
    left, right = st.columns(2)
    files_a = left.file_uploader("First group / reference images", type=["png", "jpg", "jpeg", "webp", "zip"],
                                 accept_multiple_files=True, key="image_task_a", max_upload_size=100)
    files_b = right.file_uploader("Second group / reference images", type=["png", "jpg", "jpeg", "webp", "zip"],
                                  accept_multiple_files=True, key="image_task_b", max_upload_size=100)
    positive = _read_cached(_files(files_a), "image_task_refs_a")
    negative = _read_cached(_files(files_b), "image_task_refs_b")
    for item in positive + negative:
        if item.error:
            st.error(f"Reference {item.name}: {item.error}")
            return None
    if min(len(positive), len(negative)) < 2:
        st.info("Add at least two different reference images to each group to fit a task.")
        return None
    return ImageTask(**settings, positive=tuple(item.image for item in positive),
                     negative=tuple(item.image for item in negative))


def _sheet_task():
    st.caption("Use a page with six framed examples on the left, six on the right, and two test images below.")
    uploaded = st.file_uploader("Example sheet", type=["png", "jpg", "jpeg", "webp"],
                                key="image_task_sheet", max_upload_size=100)
    if uploaded is None:
        return None, ()
    data = uploaded.getvalue()
    identity = sha256(data).hexdigest()
    cached = st.session_state.get("image_task_sheet_crops")
    if cached is None or cached[0] != identity:
        cached = identity, read_example_sheet(data)
        st.session_state.image_task_sheet_crops = cached
    positive_files, negative_files, test_files = cached[1]
    positive = _read_cached(positive_files, "image_task_refs_a")
    negative = _read_cached(negative_files, "image_task_refs_b")
    tests = _read_cached(test_files, "image_task_batch_input")
    st.success(f"Found {len(positive)} left references, {len(negative)} right references, and {len(tests)} test images.")
    with st.expander("Preview the test images", expanded=True):
        st.image(group_board([item.image for item in tests], ["Test 1", "Test 2"], columns=2), width=335)
    st.caption("Only the panel interiors are used. The frames, labels, and caption stay outside the analysis.")
    return ImageTask(**_task_settings(sheet=True), positive=tuple(item.image for item in positive),
                     negative=tuple(item.image for item in negative)), test_files


def image_tasks_workspace():
    st.subheader("Sort images by your examples")
    st.write("Teach a visual check with examples, then sort new images into two named groups.")
    st.caption("Runs locally. Uncertain images go to review. Downloads preserve individual files or the extracted sheet images.")
    st.markdown("**1. Teach the two groups**")
    source = st.radio("Start with", ["New sorting setup", "Import an example sheet", "Open saved setup"],
                      horizontal=True, key="image_task_source")
    task = None
    sheet_files = ()
    try:
        if source == "New sorting setup":
            task = _new_task()
        elif source == "Import an example sheet":
            task, sheet_files = _sheet_task()
        else:
            uploaded = st.file_uploader("Saved sorting setup", type=["zip"], key="image_task_saved", max_upload_size=100)
            if uploaded is not None:
                task = load_task(uploaded.getvalue())
                st.write(f"{task.name} · {task.groups[0]} / {task.groups[1]}")
                st.caption("Saved tasks contain reference images and settings. Fit them again using the current local solver; previous predictions are not reused.")
    except (ValueError, OSError) as error:
        st.error(str(error))
    if task is None:
        return
    stored = st.session_state.get("image_task_fitted")
    current_fit = stored is not None and stored[0] == task.fingerprint
    with st.expander("Review the examples", expanded=not current_fit):
        for column, images, label in zip(st.columns(2), (task.positive, task.negative), task.groups):
            with column:
                st.write(f"{label} · {len(images)} references")
                st.image(group_board(images, [str(i+1) for i in range(len(images))]), width="stretch")
    if not current_fit and st.button("Learn from these examples", type="primary", key="image_task_fit_button"):
        st.session_state.pop("image_task_fitted", None)
        st.session_state.pop("image_task_results", None)
        status = st.empty()
        try:
            encoder = None
            if task.representation == "photos" and not task.photo_relation:
                from .resources import semantic_encoder
                encoder = semantic_encoder()
            fitted = fit_task(task, semantic_encoder=encoder, progress=status.caption)
            st.session_state.image_task_fitted = task.fingerprint, fitted
            st.rerun()
        except (ValueError, RuntimeError, OSError, ImportError) as error:
            st.error(str(error))
        finally:
            status.empty()
    stored = st.session_state.get("image_task_fitted")
    if stored is None or stored[0] != task.fingerprint:
        st.info("Press Learn from these examples to continue. Changing the examples or group names requires a new check.")
        return
    fitted = stored[1]
    if task.photo_relation:
        condition = fitted.diagnostics["proposed_condition"]
        checks = fitted.diagnostics["reference_checks"]
        if condition:
            st.write("Proposed condition for " + task.groups[0])
            st.write(condition)
            st.caption(f"Matched {sum(check['matches'] for check in checks)} of {len(checks)} references. "
                       "These same examples helped propose the condition.")
        if not fitted.diagnostics["condition_verified"]:
            st.warning("A condition must match all the references before it can assign new images. Results will go to Needs review.")
    elif fitted.diagnostics.get("logical_explanation"):
        st.write(f"Rule for {task.groups[0]}: {fitted.diagnostics['logical_explanation']}")
        st.caption("The reference checks support this rule; new images can still reveal mistakes.")
    else:
        score = fitted.diagnostics.get("nested_cv_accuracy")
        st.caption(f"Check on your examples: {score:.0%}. This is not measured accuracy on your new images."
                   if score is not None else "Not enough references to cross-check this fit.")
        if score is None or score < .65:
            st.warning("The references do not yet establish a reliable distinction. Add more varied or more consistent examples; uncertain predictions will need review.")
    with st.expander("Save this setup for later"):
        st.download_button("Save sorting setup", task.export(), "logicapture-task.zip", "application/zip", key="image_task_save")
    if sheet_files:
        st.markdown("**2. Sort the test images**")
        st.caption("The two extracted test images are ready. The download will contain each crop as its own PNG.")
        files = sheet_files
    else:
        st.markdown("**2. Add the images to sort**")
        st.caption("Upload individual images or a ZIP, up to 100 files and 100 MB expanded. Analysis scales images to at most 1280 pixels. Reference duplicates go to review.")
        uploads = st.file_uploader("New images to sort", type=["png", "jpg", "jpeg", "webp", "zip"],
                                   accept_multiple_files=True, key="image_task_batch", max_upload_size=100)
        files = _files(uploads)
    context = task.fingerprint + ":" + _identity(files)
    try:
        items = _read_cached(files, "image_task_batch_input")
    except ValueError as error:
        st.error(str(error))
        return
    prior = st.session_state.get("image_task_results")
    have_results = prior is not None and prior["context"] == context
    if not have_results and st.button("Sort these images", type="primary", disabled=not items, key="image_task_sort"):
        st.session_state.pop("image_task_results", None)
        status = st.empty()
        try:
            rows = sort_batch(fitted, items, progress=status.caption)
            st.session_state.image_task_results = {"context": context, "rows": rows, "reviews": {}, "run": uuid4().hex[:12]}
            st.rerun()
        except (ValueError, RuntimeError, OSError, ImportError) as error:
            st.error(str(error))
        finally:
            status.empty()
    saved = st.session_state.get("image_task_results")
    if not saved or saved["context"] != context:
        return
    rows, reviews = saved["rows"], saved["reviews"]
    st.markdown("**3. Review and download**")
    labels = [None if row["error"] else reviews.get(row["id"], row["automatic_label"]) for row in rows]
    for column, label, title in zip(st.columns(3), (True, False, None), (*task.groups, "Needs review / errors")):
        column.metric(title, sum(value is label for value in labels))
    st.dataframe([{"Image": row["filename"], "Automatic": task.groups[0 if row["automatic_label"] else 1]
                  if row["automatic_label"] is not None else "Needs review",
                  "Final group": task.groups[0 if label else 1] if label is not None else "Needs review",
                  "Reviewed": row["id"] in reviews, "Reason": row["reason"]}
                 for row, label in zip(rows, labels)], hide_index=True, width="stretch")
    st.markdown("**Review or correct a decision**")
    order = sorted(range(len(rows)), key=lambda i: (labels[i] is not None, rows[i]["id"] in reviews, i))
    index = st.selectbox("Review image", order, format_func=lambda i: f"{rows[i]['id']} · {rows[i]['filename']}",
                         key="image_task_review_index_"+saved["run"])
    item, row = items[index], rows[index]
    picture, neighbors = st.columns(2)
    if item.image is not None:
        picture.image(item.image, width=min(500, item.image.shape[1]))
    if row["nearest_reference_ids"]:
        references = {f"{'A' if label else 'B'}{i+1:02d}": (image, task.groups[0 if label else 1])
                      for label, group in ((True, task.positive), (False, task.negative))
                      for i, image in enumerate(group)}
        nearest = [references[key] for key in row["nearest_reference_ids"] if key in references]
        if nearest:
            neighbors.write("Closest reference examples")
            neighbors.image(group_board([pair[0] for pair in nearest], [pair[1] for pair in nearest], tile=115), width="stretch")
    st.caption(row["reason"])
    if evidence := row["evidence"].get("visible_evidence"):
        st.write(evidence)
    if row["automatic_label"] is None and row["suggested_label"] is not None:
        st.write("Suggestion for your review: " + task.groups[0 if row["suggested_label"] else 1])
        st.caption("This suggestion did not pass the automatic checks. It stays in Needs review unless you assign a group.")
    if not row["error"]:
        options = ["Use automatic decision", "Group 1 · "+task.groups[0], "Group 2 · "+task.groups[1], "Needs review"]
        choice = st.radio("Label for this image", options, horizontal=True,
                          key=f"image_task_review_{saved['run']}_{row['id']}")
        if st.button("Apply label", key="image_task_review_save"):
            if choice == options[0]:
                reviews.pop(row["id"], None)
            else:
                reviews[row["id"]] = True if choice == options[1] else False if choice == options[2] else None
            st.rerun()
    st.caption("Review changes the exported group. It does not retrain the task or turn a reviewed image into independent validation.")
    export_key = sha256((context+json.dumps(reviews, sort_keys=True)).encode()).hexdigest()
    if saved.get("export_key") != export_key:
        saved["export"] = export_results(task, fitted, items, rows, reviews)
        saved["export_key"] = export_key
    st.download_button("Download sorted images + report", saved["export"], "logicapture-sorted.zip", "application/zip", key="image_task_export", type="primary")
    with st.expander("Task evidence"):
        st.json(fitted.diagnostics, expanded=False)
