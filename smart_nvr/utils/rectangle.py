class Rectangle:
    # topleft
    x1: int
    y1: int

    # bottomright
    x2: int
    y2: int

    def __init__(self, x1: int, y1: int, x2: int, y2: int) -> None:
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    def overlap_area(self, other: "Rectangle") -> int:
        dx = min(self.x2, other.x2) - max(self.x1, other.x1)
        dy = min(self.y2, other.y2) - max(self.y1, other.y1)

        if (dx >= 0) and (dy >= 0):
            return dx * dy

        return 0

    def __repr__(self) -> str:
        kws = [f"{key}={value!r}" for key, value in self.__dict__.items()]
        return f"{self.__class__.__name__}({', '.join(kws)})"
