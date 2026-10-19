"""Local camera state machine: detect, stabilize, infer off-thread, verify, draw.

No Streamlit calls run on media threads. Changed/lost pages invalidate old
results, including late worker completions. Only one inference job is queued.
"""
from concurrent.futures import ThreadPoolExecutor
from threading import RLock
from time import monotonic
import textwrap

import cv2
import numpy as np

from .panel_detection import split_problem_page, panel_overlay


def analyze_live_panels(panels):
    from src.reasoning.drawing import fit_drawing_rules
    from src.reasoning.examples import generate_checked_examples
    from src.active import GenerationError
    positive = [panel.image for panel in panels if panel.label]
    negative = [panel.image for panel in panels if not panel.label]
    fit, attempt, pipeline = fit_drawing_rules(positive,negative)
    result = {"pipeline":pipeline,"structural_attempt":attempt,"examples":[],"positive":positive,"negative":negative}
    if fit is None:
        result["message"] = "No support-checked executable rule. Capture this page for broader analysis."
        return result
    result["message"] = attempt["candidates"][0]["text"]
    try:
        result["examples"] = generate_checked_examples({"structural_attempt":attempt},attempts=24)
    except GenerationError as error:
        result["generation_error"] = str(error)
    return result


class LivePageEngine:
    def __init__(self, *, analyzer=analyze_live_panels, interval=.45, stable_samples=3, executor=None):
        self.lock = RLock()
        self.executor = executor or ThreadPoolExecutor(max_workers=1,thread_name_prefix="bongard-live")
        self.analyzer, self.interval, self.stable_samples = analyzer,interval,stable_samples
        self.epoch = 0
        self.pending = None
        self.pending_epoch = None
        self.result = None
        self.signature = None
        self.samples = 0
        self.last_sample = -float("inf")
        self.page = None
        self.panels = []
        self.attempted = False
        self.attempt_count = 0
        self.last_attempt = -float("inf")
        self.message = "Point the camera at the whole problem. Positives left; negatives right."

    def reset(self):
        with self.lock:
            self.epoch += 1
            self.result = self.signature = self.page = None
            self.panels = []
            self.samples = 0
            self.attempted = False
            self.attempt_count = 0
            self.last_sample = -float("inf")
            self.message = "Camera stopped."

    def close(self):
        self.reset()
        self.executor.shutdown(wait=False,cancel_futures=True)

    def snapshot(self):
        with self.lock:
            return {"message":self.message,"result":self.result,"panels":tuple(self.panels),"epoch":self.epoch,
                    "processing":self.pending is not None,"stable_samples":self.samples}

    @staticmethod
    def _signature(panels):
        # Each crop is normalized separately: camera translation and page-scale
        # changes do not cause repeated fitting. Thresholded ink rejects lighting noise.
        signatures = []
        for panel in panels:
            gray = cv2.cvtColor(cv2.resize(panel.image,(48,48)),cv2.COLOR_RGB2GRAY)
            signatures.append(cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1] > 0)
        return np.stack(signatures)

    @staticmethod
    def _changed(a,b):
        if a is None or a.shape != b.shape:
            return True
        # Normalize by ink, not the mostly-white page: otherwise a completely
        # different sparse drawing could retain the previous page's rule.
        kernel = np.ones((3,3),np.uint8)
        dilated_a = np.stack([cv2.dilate(v.astype(np.uint8),kernel)>0 for v in a])
        dilated_b = np.stack([cv2.dilate(v.astype(np.uint8),kernel)>0 for v in b])
        unmatched = np.count_nonzero(a & ~dilated_b)+np.count_nonzero(b & ~dilated_a)
        return unmatched/max(1,np.count_nonzero(a)+np.count_nonzero(b)) > .12

    def process(self, image, *, now=None):
        now = monotonic() if now is None else now
        with self.lock:
            if now-self.last_sample >= self.interval:
                self.last_sample = now
                try:
                    page, panels, diagnostic = split_problem_page(image)
                    signature = self._signature(panels)
                    changed = self._changed(self.signature,signature)
                    if changed:
                        self.epoch += 1
                        self.result = None
                        self.attempted = False
                        self.attempt_count = 0
                        self.signature = signature
                        self.samples = 1
                    else:
                        self.samples += 1
                    self.page, self.panels = page,panels
                    self.message = f"Detected {diagnostic['positive']} positive / {diagnostic['negative']} negative panels. Hold still."
                except (ValueError,cv2.error) as error:
                    self.epoch += 1
                    self.signature = self.result = self.page = None
                    self.samples = 0
                    self.panels = []
                    self.attempted = False
                    self.attempt_count = 0
                    self.message = str(error)
            if self.pending is not None and self.pending.done():
                try:
                    result = self.pending.result()
                    if self.pending_epoch == self.epoch:
                        self.result = result
                except Exception as error:
                    if self.pending_epoch == self.epoch:
                        self.message = "Local analysis failed: "+str(error)
                finally:
                    self.pending = None
            failed_reading = (self.result is not None and not self.result.get("structural_attempt",{}).get("accepted",False))
            retry = failed_reading and self.attempt_count < 3 and now-self.last_attempt >= 2.
            if self.panels and self.samples >= self.stable_samples and (not self.attempted or retry) and self.pending is None:
                self.pending_epoch = self.epoch
                self.pending = self.executor.submit(self.analyzer,tuple(self.panels))
                self.attempted = True
                self.attempt_count += 1
                self.last_attempt = now
            if self.result is not None:
                self.message = self.result["message"]
            elif self.pending is not None and self.pending_epoch == self.epoch:
                self.message = "Checking a stable page locally..."
            return self._render(image)

    def _render(self, frame):
        top = panel_overlay(self.page,self.panels) if self.page is not None else frame.copy()
        if max(top.shape[:2]) > 1280:
            h,w = top.shape[:2]
            top = cv2.resize(top,(round(w*1280/max(h,w)),round(h*1280/max(h,w))))
        # The evidence and generated pair are part of the returned live video.
        width = max(640,top.shape[1])
        output = np.full((top.shape[0]+250,width,3),255,np.uint8)
        output[:top.shape[0],:top.shape[1]] = top
        y = top.shape[0]+22
        for line in textwrap.wrap(self.message,max(40,width//9))[:2]:
            cv2.putText(output,line,(10,y),cv2.FONT_HERSHEY_SIMPLEX,.46,(0,0,0),1,cv2.LINE_AA)
            y += 19
        if self.result is not None:
            examples = self.result.get("examples",[])
            for i,example in enumerate(examples):
                x = 12+i*(width//2)
                thumb = cv2.resize(example.image,(160,160))
                output[y+22:y+182,x:x+160] = thumb
                label = "A / new positive" if example.label else "B / new negative"
                cv2.putText(output,label,(x,y+15),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,47,167),1,cv2.LINE_AA)
            if not examples:
                cv2.putText(output,"Checked example generation unavailable",(12,y+30),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1)
        return output
