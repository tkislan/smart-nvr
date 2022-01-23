from typing import List, Tuple

from .rectangle import Rectangle


def square_box_rectangles(
    rectangle: Rectangle,
    shape: Tuple[int, int],  # height, width
) -> Rectangle:
    img_height, img_width = shape

    if rectangle.width < 300 and rectangle.height < 300:
        size = 300
    elif rectangle.width < 600 and rectangle.height < 600:
        size = 600
    elif rectangle.width < 900 and rectangle.height < 900:
        size = 900

    topleft_x = max(0, rectangle.x1 - int((size - rectangle.width) / 2))
    topleft_y = max(0, rectangle.y1 - int((size - rectangle.height) / 2))

    bottomright_x = min(
        img_width - 1,
        (rectangle.x1 + rectangle.width) + int((size - rectangle.width) / 2),
    )
    bottomright_y = min(
        img_height - 1,
        (rectangle.y1 + rectangle.height) + int((size - rectangle.height) / 2),
    )

    topleft_x = max(0, bottomright_x - size)
    topleft_y = max(0, bottomright_y - size)

    bottomright_x = topleft_x + size
    bottomright_y = topleft_y + size

    return Rectangle(topleft_x, topleft_y, bottomright_x, bottomright_y)


def outer_box_rectangles(rectangles: List[Rectangle]) -> Rectangle:
    return Rectangle(
        min([r.x1 for r in rectangles]),
        min([r.y1 for r in rectangles]),
        min([r.x2 for r in rectangles]),
        min([r.y2 for r in rectangles]),
    )


def merge_rectangles(
    rectangles: List[Rectangle], overlap_ratio_threshold: float = 0.65
) -> List[Rectangle]:
    rectangle_groups: List[List[Rectangle]] = []

    for rectangle in rectangles:
        rectangle_grouped = False
        for rectangle_group in rectangle_groups:
            for other_rectangle in rectangle_group:
                smaller_area = min(rectangle.area, other_rectangle.area)
                if (
                    rectangle.overlap_area(other_rectangle)
                    > overlap_ratio_threshold * smaller_area
                ):
                    rectangle_group.append(rectangle)
                    rectangle_grouped = True
                    break

            if rectangle_grouped:
                break

        if not rectangle_grouped:
            rectangle_groups.append([rectangle])

        rectangle_groups = sorted(
            rectangle_groups, key=lambda rectangle_group: len(rectangle_group)
        )

    return [
        outer_box_rectangles(rectangle_group) for rectangle_group in rectangle_groups
    ]
