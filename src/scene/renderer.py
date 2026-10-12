from io import BytesIO
from PIL import Image, ImageDraw

from .objects import Scene

PALETTE = {"black": "#000000", "red": "#d95751", "blue": "#002FA7", "green": "#309676"}


def draw_scene(scene: Scene, size=320, stroke=3, highlight=(), relation_lines=False):
    scale = 2
    image = Image.new("RGB", (size*scale, size*scale), "white")
    draw = ImageDraw.Draw(image)
    for obj in sorted(scene.objects, key=lambda o: -o.area):
        color = PALETTE.get(obj.color, PALETTE["black"])
        if obj.id in highlight: color = "#002FA7"
        if obj.kind == "path" and obj.path is not None:
            path = obj.path
            factor = size*scale
            for segment in path.segments or (path.points,):
                points = [(round(x*factor), round(y*factor)) for x, y in segment]
                if not path.segments and path.topology == "closed" and points[0] != points[-1]: points.append(points[0])
                draw.line(points, fill=color, width=max(stroke*scale, round(path.width*factor)), joint="curve")
            for arrow in path.arrows:
                if arrow.direction == 0: continue
                # Taper along the actual centerline. A straight triangle placed
                # across a tight bend can leave its tip detached from the path.
                # These are marks on one curve, not triangle-shaped entities.
                left,right=[],[]
                for step in range(19):
                    fraction=step/18
                    position=arrow.arc_position+arrow.direction*(-.030+.072*fraction)/max(path.length,1e-9)
                    point,tangent=path.point_tangent(position)
                    halfwidth=.025*(1-fraction)
                    nx,ny=-tangent[1]*halfwidth,tangent[0]*halfwidth
                    left.append(((point[0]+nx)*factor,(point[1]+ny)*factor))
                    right.append(((point[0]-nx)*factor,(point[1]-ny)*factor))
                draw.polygon(left+list(reversed(right)),fill=color)
            if relation_lines: draw.text((obj.center_x*factor, obj.center_y*factor), str(obj.id), fill="#002FA7")
            continue
        if obj.kind == "unknown" or obj.shape == "unknown":
            if obj.contour:
                points = [(round(x*size*scale), round(y*size*scale)) for x,y in obj.contour]
                if obj.kind == "region":
                    draw.polygon(points, fill=color if obj.filled else "white")
                    draw.line(points+[points[0]], fill=color, width=stroke*scale, joint="curve")
                else:
                    draw.line(points, fill=color, width=stroke*scale, joint="curve")
            continue
        points = [(round(x*size*scale), round(y*size*scale)) for x, y in obj.vertices()]
        draw.polygon(points, fill=color if obj.filled else "white")
        draw.line(points + [points[0]], fill=color, width=stroke*scale, joint="curve")
        if obj.id in highlight:
            draw.line(points + [points[0]], fill="#002FA7", width=(stroke+2)*scale, joint="curve")
        if relation_lines:
            draw.text((obj.center_x*size*scale, obj.center_y*size*scale), str(obj.id), fill="#002FA7")
    return image.resize((size, size), Image.Resampling.LANCZOS)


def png_bytes(image):
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def draw_page(groups, panel_size=220):
    """Fixed controlled template: A in columns 0–1, B in columns 2–3."""
    if any(len(g) != 6 for g in groups):
        raise ValueError("The controlled page requires six panels per group")
    gap, top = 18, 64
    width, height = 4*panel_size+5*gap, 3*panel_size+4*gap+top
    page = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(page)
    draw.text((gap, 24), "GROUP A", fill="black")
    draw.text((3*gap+2*panel_size, 24), "GROUP B", fill="black")
    for group_index, group in enumerate(groups):
        for i, scene in enumerate(group):
            col, row = 2*group_index+i%2, i//2
            x, y = gap+col*(panel_size+gap), top+gap+row*(panel_size+gap)
            page.paste(draw_scene(scene, panel_size-12), (x+6, y+6))
            draw.rectangle((x, y, x+panel_size, y+panel_size), outline="black", width=2)
    return page
