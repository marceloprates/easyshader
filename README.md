# easyshader: Examples

easyshader is a tool for rendering 3D scenes, exporting .ply files for 3D printing and creating animations, powered by Signed Distance Fields (SDFs) and written in Python/Taichi.

It was created to enable drawing 3D shapes using a very concise syntax, and is packed with 3D primitives, transformations and smooth operators.

[![Run in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marceloprates/easyshader/blob/main/README.ipynb)

<img src="pictures/logo.png" width="25%">


# Basic usage


```python
from easyshader import *
```

```python
Sphere(1)
```


    
![png](README_files/README_4_0.png)
    


Use the "color" parameter to paint your object


```python
Icosahedron(1,'#f55')
```


    
![png](README_files/README_6_0.png)
    


# easyshader primitives

You can choose from the following primitives:

- Box
- BoxFrame
- Callable
- Cone
- Cyllinder
- Icosahedron
- Iterable
- Line
- Number
- Octahedron
- Shape
- Sphere
- Torus


```python
for obj in [Sphere(1), Cyllinder(1,1), Cone(1,2), Torus(1,.2), Box(1), BoxFrame(1,.1), Icosahedron(1), Octahedron(1)]:
    display(obj.paint('#0B4F6C'))
```


    
![png](README_files/README_9_0.png)
    



    
![png](README_files/README_9_1.png)
    



    
![png](README_files/README_9_2.png)
    



    
![png](README_files/README_9_3.png)
    



    
![png](README_files/README_9_4.png)
    



    
![png](README_files/README_9_5.png)
    



    
![png](README_files/README_9_6.png)
    



    
![png](README_files/README_9_7.png)
    


# Exporting to .ply for usage in Blender or 3D printing

Export your creations to polygon meshes for 3d printing or rendering on external apps (e.g. Blender)


```python
#Icosahedron(1).to_mesh(simplify = 20, save_path='icosahedron.ply')
```

Color your creations using functions defined over x,y,z and a color palette:


```python
palette = ['#B80C09','#0B4F6C','#01BAEF','#FBFBFF','#040F16']

x = Box(.8,'palette(6*(x+y+z))',palette = palette)
x = x.isometric()

x
```


    
![png](README_files/README_14_0.png)
    


# Binary operations

## Union:


```python
BoxFrame(1,.1,'#0B4F6C') + Sphere(.5,'#B80C09')
```


    
![png](README_files/README_17_0.png)
    


## Difference:


```python
Box(1,'#0B4F6C') - Sphere(1.2)
```


    
![png](README_files/README_19_0.png)
    


## Intersection:


```python
Icosahedron(1,'#0B4F6C') & Sphere(1.1)
```


    
![png](README_files/README_21_0.png)
    


# Examples

## A coffee cup!


```python
x = Sphere(1, 'palette(6*(x+y+z))', palette = palette)
x = x.twist(4)

x &= Cyllinder(.5,.5)
x -= Cyllinder(.4,.5) + 'dy .1'
x += (Torus(.3,.05) & Shape('-x')) + 'dx .5'
x = x.isometric()
x += 'rx -pi/3'

x
```


    
![png](README_files/README_24_0.png)
    


# Create videos!
Use the 't' (time) variable to control the animation


```python
x = BoxFrame(1,.1,'palette(6*t + 6*(x+y+z))',palette = palette)
x += '.1*sin(t)'
x += 'ry t'
x += 'rx t'

x.animate(frames = 60, framerate = 15, iterations = 1000)
```

    Animating..: 100%|██████████| 59/59 [08:59<00:00,  9.14s/it]





    
![png](README_files/README_26_1.png)
    



# Transformations

## Translation


```python
Sphere(1) + 'dx .5'
```


    
![png](README_files/README_29_0.png)
    



```python
Sphere(1) + '(.1,.2,.3)'
```


    
![png](README_files/README_30_0.png)
    


## Scale


```python
Sphere(1) * .2
```


    
![png](README_files/README_32_0.png)
    



```python
Sphere(1) * (1,.2,1)
```


    
![png](README_files/README_33_0.png)
    


## Rotation


```python
Box(1) + 'rx pi/4'
```


    
![png](README_files/README_35_0.png)
    


## Advanced transformations

You can use x,y,z (and the time parameter, t) as variables in transformations such as translation, rotation, scale


```python
BoxFrame(1,.1,'#f44') + 'dx .2*y'
```


    
![png](README_files/README_38_0.png)
    


# Other operations

## Twist along an axis


```python
Box(1,'#f44').twist(2,'y')
```


    
![png](README_files/README_41_0.png)
    


## Create an "onion" shape


```python
# Create an onion from a box
x = Box(1,'#f44').onion()
# Cut a hole in the onion
x &= Shape('z')

x
```


    
![png](README_files/README_43_0.png)
    


# Smooth operators

## Smooth union


```python
sphere = (Sphere(.5,'#f44') + 'dx -.5')
box = (Box(.5,'#4ff') + 'dx +.5')

# Normal union
display(sphere + box)

# Smooth union
display(sphere <<su(.5)>> box)
```


    
![png](README_files/README_46_0.png)
    



    
![png](README_files/README_46_1.png)
    


## Smooth difference


```python
sphere = Sphere(1.1)
box = Box(1,'#f44')

# Normal difference
display(box - sphere)

# Smooth difference
display(box <<sd(.5)>> sphere)
```


    
![png](README_files/README_48_0.png)
    



    
![png](README_files/README_48_1.png)
    


## Smooth intersection


```python
sphere = Sphere(1)
box = Box(.9,'#f44')

# Normal intersection
display(box & sphere)

# Smooth intersection
display(box <<si(.5)>> sphere)
```


    
![png](README_files/README_50_0.png)
    



    
![png](README_files/README_50_1.png)
    


# Use custom taichi functions!


```python
'''
from easyshader import *

@ti.func
def mandelbulb_fn(p,max_it,k):
    z, dr, r = p, 1., 0.
    for i in range(max_it):
        r, steps = z.norm(), i
        if r > 4.0: break;
        # convert to polar coordinates
        theta = ti.acos(z.z/r)
        phi = ti.atan2(z.y,z.x)
        dr = r**2 * 3 * dr + 1.0
        # scale and rotate the point
        zr = r**3
        theta = theta*3
        phi = phi*3
        # convert back to cartesian coordinates
        z = zr*ti.Vector([
            ti.sin(theta)*ti.cos(phi),
            ti.sin(phi)*ti.sin(theta),
            ti.cos(theta)
        ])
        z+=p
    out = ti.log(r)*r/dr
    if k==1:
        out = r
    return out

# Create mandelbulb shape
mandelbulb = Shape(
    # Call 'mandelbulb_fn' to compute the SDF
    sdf = '.2*mandelbulb_fn(p,10,0)',
    #color = 'palette(200*mandelbulb_fn(1.1*p))',
    color = 'palette(12*mandelbulb_fn(p,10,1))',
    palette = ['#0C0F0A', '#FBFF12', '#FF206E', '#41EAD4', '#FFFFFF'],
    # Pass 'mandelbulb_fn' as a keyword argument
    mandelbulb_fn = mandelbulb_fn,
)

mandelbulb = mandelbulb.isometric()

mandelbulb.render(
    resolution = (2480,3508),
    fov = .25,
    max_raymarch_steps = 200,
    iterations = 1000,
        verbose = True
)'''
```




    "\nfrom easyshader import *\n\n@ti.func\ndef mandelbulb_fn(p,max_it,k):\n    z, dr, r = p, 1., 0.\n    for i in range(max_it):\n        r, steps = z.norm(), i\n        if r > 4.0: break;\n        # convert to polar coordinates\n        theta = ti.acos(z.z/r)\n        phi = ti.atan2(z.y,z.x)\n        dr = r**2 * 3 * dr + 1.0\n        # scale and rotate the point\n        zr = r**3\n        theta = theta*3\n        phi = phi*3\n        # convert back to cartesian coordinates\n        z = zr*ti.Vector([\n            ti.sin(theta)*ti.cos(phi),\n            ti.sin(phi)*ti.sin(theta),\n            ti.cos(theta)\n        ])\n        z+=p\n    out = ti.log(r)*r/dr\n    if k==1:\n        out = r\n    return out\n\n# Create mandelbulb shape\nmandelbulb = Shape(\n    # Call 'mandelbulb_fn' to compute the SDF\n    sdf = '.2*mandelbulb_fn(p,10,0)',\n    #color = 'palette(200*mandelbulb_fn(1.1*p))',\n    color = 'palette(12*mandelbulb_fn(p,10,1))',\n    palette = ['#0C0F0A', '#FBFF12', '#FF206E', '#41EAD4', '#FFFFFF'],\n    # Pass 'mandelbulb_fn' as a keyword argument\n    mandelbulb_fn = mandelbulb_fn,\n)\n\nmandelbulb = mandelbulb.isometric()\n\nmandelbulb.render(\n    resolution = (2480,3508),\n    fov = .25,\n    max_raymarch_steps = 200,\n    iterations = 1000,\n        verbose = True\n)"




```python

```
