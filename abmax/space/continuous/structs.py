from flax import struct
import jax
import jax.numpy as jnp

@struct.dataclass
class Point:
    """
    A point in 2D space.
    """
    x: jnp.float32
    y: jnp.float32

@struct.dataclass
class Line:
    """
    A line segment defined by two points.
    """
    p1: Point
    p2: Point

@struct.dataclass
class Circle:
    """
    A circle defined by its center:Point and radius.
    """
    center: Point
    radius: jnp.float32

@struct.dataclass
class Ray:
    """
    A ray defined by its origin:Point, direction:Point and length.
    direction is a unit vector (cos, sin)
    """
    origin: Point
    direction: Point # cos, sin
    length: float

# creating rays spontaneously uses jnp.arange which cannot be jit compiled thus we need to define these constants globally
RAY_RESOLUTION = 11 # number of rays
RAY_SPAN = jnp.pi/6
RAY_MAX_LENGTH = 40.0

@struct.dataclass
class Space:
    """
    A space defined by its bounds and walls.
    torous: bool determines if the space is torous or not i.e. exiting one side of the space will enter from the other side
    walls: Line is a vmapped array of walls in the space, A wall is defined by two points
    """
    x_min: jnp.float32
    x_max: jnp.float32
    y_min: jnp.float32
    y_max: jnp.float32
    torous: bool
    walls: Line # vmapped