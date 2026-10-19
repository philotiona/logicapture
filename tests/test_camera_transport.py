"""Optional real local WebRTC round trip; no webcam, hosted ICE, or API calls."""
import asyncio
from pathlib import Path

import numpy as np
from PIL import Image
import pytest


def test_camera_frames_return_with_detected_page_and_checked_examples(tmp_path):
    rtc = pytest.importorskip("aiortc")
    av = pytest.importorskip("av")
    from src.vision.live import LivePageEngine
    page = np.array(Image.open(Path(__file__).resolve().parents[1]/"data/fixtures/dotted_page.png").convert("RGB"))

    async def round_trip():
        browser = rtc.RTCPeerConnection(rtc.RTCConfiguration(iceServers=[]))
        server = rtc.RTCPeerConnection(rtc.RTCConfiguration(iceServers=[]))
        engine = LivePageEngine(interval=.1,stable_samples=2)
        done = asyncio.Event()
        tasks,received = [],[]

        class Camera(rtc.VideoStreamTrack):
            async def recv(self):
                pts,time_base = await self.next_timestamp()
                frame = av.VideoFrame.from_ndarray(page,format="rgb24")
                frame.pts,frame.time_base = pts,time_base
                return frame

        class Processed(rtc.VideoStreamTrack):
            def __init__(self,track):
                super().__init__(); self.track=track
            async def recv(self):
                frame = await self.track.recv()
                output = av.VideoFrame.from_ndarray(engine.process(frame.to_ndarray(format="rgb24")),format="rgb24")
                output.pts,output.time_base = frame.pts,frame.time_base
                return output

        @server.on("track")
        def process(track):
            server.addTrack(Processed(track))

        @browser.on("track")
        def receive(track):
            async def collect():
                while True:
                    frame = await track.recv()
                    received.append((frame.width,frame.height))
                    result = engine.snapshot()["result"]
                    if result and len(result["examples"])==2:
                        done.set()
                        break
            tasks.append(asyncio.create_task(collect()))

        try:
            browser.addTrack(Camera())
            await browser.setLocalDescription(await browser.createOffer())
            await server.setRemoteDescription(browser.localDescription)
            await server.setLocalDescription(await server.createAnswer())
            await browser.setRemoteDescription(server.localDescription)
            try:
                await asyncio.wait_for(done.wait(),timeout=20)
            except TimeoutError as error:
                state = engine.snapshot()
                result = state["result"]
                from src.vision.observations import observations
                if result:
                    for i,image in enumerate(result.get("positive",[])+result.get("negative",[])):
                        Image.fromarray(image).save(tmp_path/f"codec_support_{i}.png")
                raise AssertionError({"browser":browser.connectionState,"server":server.connectionState,
                    "received":len(received),"message":state["message"],"panels":len(state["panels"]),
                    "samples":state["stable_samples"],"result":{k:v for k,v in (result or {}).items() if k not in ("positive","negative","examples")},
                    "dot_spacing_p90":[observations(image)["dot_spacing_p90"] for image in (result or {}).get("positive",[])+(result or {}).get("negative",[])],
                    "task_errors":[str(t.exception()) for t in tasks if t.done() and not t.cancelled()]}) from error
            assert received[-1][1] > page.shape[0]
            result = engine.snapshot()["result"]
            assert result["structural_attempt"]["accepted"]
            assert all(e.verification["pixel_reread"] for e in result["examples"])
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)
            await browser.close()
            await server.close()
            engine.close()

    asyncio.run(round_trip())
