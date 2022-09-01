import time
import threading
try:
    from greenlet import getcurrent as get_ident
except ImportError:
    try:
        from thread import get_ident
    except ImportError:
        from _thread import get_ident


class CameraEvent(object):
    """An Event-like class that signals all active clients when a new frame is
    available.
    一个类似事件的类，当一个新的帧可用时向所有活动的客户端发出信号。
    """
    def __init__(self):
        self.events = {}

    def wait(self):
        """Invoked from each client's thread to wait for the next frame.
        从每个客户端的线程调用以等待下一帧
        """

        ident = get_ident()
        print("wait")
        if ident not in self.events:
            # this is a new client
            # add an entry for it in the self.events dict
            # each entry has two elements, a threading.Event() and a timestamp
            # 这是一个新客户
            # 在self.events字典中为它添加一个条目
            # 每个条目有两个元素，一个线程。事件和时间戳
            self.events[ident] = [threading.Event(), time.time()]
        return self.events[ident][0].wait()

    def set(self):
        """Invoked by the camera thread when a new frame is available.
        当新帧可用时，由camera线程调用
        """
        now = time.time()
        remove = None
        for ident, event in self.events.items():
            if not event[0].isSet():
                # if this client's event is not set, then set it
                # also update the last set timestamp to now
                # 如果此客户端的事件未设置，则设置它
                # 还将上次设置的时间戳更新为现在
                event[0].set()
                event[1] = now
            else:
                # if the client's event is already set, it means the client
                # did not process a previous frame
                # if the event stays set for more than 5 seconds, then assume
                # the client is gone and remove it
                # 如果已经设置了客户端的事件，则表示客户端
                # 未处理前一帧
                # 如果事件保持设置超过5秒，则假设
                # 客户端已不在，请将其移除
                if now - event[1] > 5:
                    remove = ident
        if remove:
            del self.events[remove]

    def clear(self):
        """Invoked from each client's thread after a frame was processed.
        在处理帧后从每个客户端的线程调用。

        """

        self.events[get_ident()][0].clear()


class BaseCamera(object):
    thread = None  # background thread that reads frames from camera #从相机读取帧的后台线程
    frame = None  # current frame is stored here by background thread 当前帧由后台线程存储在这里
    last_access = 0  # time of last client access to the camera   客户端最后一次访问摄像机的时间
    event = CameraEvent()

    def __init__(self):
        """Start the background camera thread if it isn't running yet."""

        " " "如果后台摄像头线程尚未运行，请启动它。"""

        if BaseCamera.thread is None:
            BaseCamera.last_access = time.time()

            # start background frame thread #开始后台框架线程
            BaseCamera.thread = threading.Thread(target=self._thread)
            BaseCamera.thread.start()

            # wait until frames are available
            #等待，直到帧可用
            while self.get_frame() is None:
                time.sleep(0)

    def get_frame(self):
        """Return the current camera frame."""
        #返回当前相机帧

        BaseCamera.last_access = time.time()

        # 等待来自相机线程的信号
        BaseCamera.event.wait()
        BaseCamera.event.clear()


        return BaseCamera.frame

    @staticmethod
    def frames(source,weights):
        """"Generator that returns frames from the camera."""
        raise RuntimeError('Must be implemented by subclasses.')

    @classmethod
    def _thread(cls):
        """Camera background thread."""
        print('Starting camera thread.')

        frames_iterator = cls.frames('yolov5s.pt','https://open.ys7.com/v3/openlive/G18183870_1_2.m3u8?expire=1677567819&id=420234950509830144&t=c784a19375d066e715d175a1d68ad8146b88a84447e03c7bb751f02c43aceb0b&ev=100')
        # print(cls.frames('yolov5s.pt',
        #                              'https://open.ys7.com/v3/openlive/G18183870_1_2.m3u8?expire=1677567819&id=420234950509830144&t=c784a19375d066e715d175a1d68ad8146b88a84447e03c7bb751f02c43aceb0b&ev=100'))

        print('frames_iterator... {}, type: {}'.format(frames_iterator, type(frames_iterator)))
        for frame in frames_iterator:
            BaseCamera.frame = frame
            BaseCamera.event.set()  # send signal to clients
            time.sleep(0)


            # if there hasn't been any clients asking for frames in
            # the last 10 seconds then stop the thread
            if time.time() - BaseCamera.last_access > 10:
                frames_iterator.close()
                print('Stopping camera thread due to inactivity.')
                break
        BaseCamera.thread = None
