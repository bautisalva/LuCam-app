
def lucam_test():
    """Robust Lucam test with multiple functionalities."""
    import time
    import numpy as np
    from lucam1 import Lucam  # adjust if needed

    try:
        # === Initialize Camera ===
        lucam = Lucam(1)
        print('Camera initialized.')
        print('Camera Properties:', lucam)

        # === Safe Format ===
        lucam.SetFormat(
            Lucam.FrameFormat(
                x=0,
                y=0,
                width=640,
                height=480,
                binningX=1,
                flagsX=1,
                binningY=1,
                flagsY=1,
            ),
            framerate=10.0,
        )
        print('Format set.')

        # === Set Safe Camera Properties ===
        lucam.set_properties(
            brightness=1.0,
            contrast=1.0,
            saturation=1.0,
            hue=0.0,
            gamma=1.0,
            exposure=100.0,
            gain=1.0,
        )
        print('Camera properties set.')

        # === Take Snapshot and Save ===
        image = lucam.TakeSnapshot()
        np.fill_diagonal(image, 255)
        lucam.SaveImage(image, 'safe_snapshot.tif')
        print('Saved safe_snapshot.tif')

        # === Fast Frame Snapshots ===
        snapshot = Lucam.Snapshot(
            exposure=lucam.exposure,
            gain=1.0,
            timeout=1000.0,
            format=lucam.GetFormat()[0]
        )
        lucam.EnableFastFrames(snapshot)
        for i in range(3):
            lucam.TakeFastFrame(image, validate=False)
            filename = f'fast_frame_{i + 1}.tif'
            lucam.SaveImage(image, filename)
            print(f'Saved {filename}')
        lucam.DisableFastFrames()

        # === Stream and Record Video ===
        lucam.StreamVideoControl('start_streaming')
        video = lucam.TakeVideo(10)
        lucam.StreamVideoControl('stop_streaming')
        lucam.SaveImage(video, 'video_frames.raw')
        print('Saved video_frames.raw')

        # === Display Window with Framerate ===
        lucam.CreateDisplayWindow(b'DisplayTest')
        lucam.StreamVideoControl('start_display')
        time.sleep(1.0)
        lucam.AdjustDisplayWindow(
            width=lucam.GetFormat()[0].width * 2,
            height=lucam.GetFormat()[0].height * 2
        )
        time.sleep(1.0)
        rate = lucam.QueryDisplayFrameRate()
        print(f'Display framerate: {rate:.2f} FPS')
        lucam.StreamVideoControl('stop_streaming')
        lucam.DestroyDisplayWindow()

        # === Read Non-volatile Memory ===
        memory = lucam.PermanentBufferRead()
        print('Non-volatile memory:', memory)

        # === Reset and Switch to New Format ===
        lucam.CameraReset()
        lucam.SetFormat(
            Lucam.FrameFormat(
                x=0,
                y=0,
                width=320,
                height=240,
                binningX=1,
                flagsX=1,
                binningY=1,
                flagsY=1,
            ),
            framerate=5.0,
        )
        print('Camera reset and format changed.')

        # === Final Snapshot After Reset ===
        image = lucam.TakeSnapshot()
        lucam.SaveImage(image, 'post_reset_snapshot.tif')
        print('Saved post_reset_snapshot.tif')

    except Exception as e:
        print(f'Error: {e}')

    finally:
        try:
            lucam.CameraClose()
            del lucam
            print('Camera closed.')
        except:
            pass


if __name__ == '__main__':
    lucam_test()
