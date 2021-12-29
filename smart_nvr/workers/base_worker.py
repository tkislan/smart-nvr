import threading


class BaseWorker(threading.Thread):
    def __init__(self, name: str = None):
        super().__init__(name=name)
        self._should_exit = threading.Event()

    def run_processing(self):
        raise NotImplementedError()

    def teardown(self):
        pass

    def run(self):
        while not self._should_exit.is_set():
            try:
                self.run_processing()
            except Exception as error:
                print(f"{self.__class__.__name__} processing failed", error)
        self.teardown()

    def stop(self):
        print(f"Stopping {self.__class__.__name__}")
        self._should_exit.set()
