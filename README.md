# easyshader: Examples


easyshader is a tool for rendering 3D scenes, exporting .ply files for 3D printing and creating animations, powered by Signed Distance Fields (SDFs) and written in Python/Taichi.

![](pictures/logo.png)

It was created to enable drawing 3D shapes using a very concise syntax, and is packed with 3D primitives, transformations and smooth operators.

# Basic usage


```python
from easyshader import *

Sphere(1)
```


    
![png](README_files/README_3_0.png)
    


Use the "color" parameter to paint your object


```python
Icosahedron(1,'#f55')
```


    
![png](README_files/README_5_0.png)
    


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


    
![png](README_files/README_8_0.png)
    



    
![png](README_files/README_8_1.png)
    



    
![png](README_files/README_8_2.png)
    



    
![png](README_files/README_8_3.png)
    



    
![png](README_files/README_8_4.png)
    



    
![png](README_files/README_8_5.png)
    



    
![png](README_files/README_8_6.png)
    



    
![png](README_files/README_8_7.png)
    


# Exporting to .ply for usage in Blender or 3D printing

Export your creations to polygon meshes for 3d printing or rendering on external apps (e.g. Blender)


```python
Icosahedron(1).to_mesh(simplify = 20, save_path='icosahedron.ply')
```

Color your creations using functions defined over x,y,z and a color palette:


```python
palette = ['#B80C09','#0B4F6C','#01BAEF','#FBFBFF','#040F16']

x = Box(.9, 'palette(4*(x+y**2+z**3))', palette = palette)
x = x.isometric()
x
```


    
![png](README_files/README_13_0.png)
    


# Binary operations

## Union:


```python
BoxFrame(1,.1,'#0B4F6C') + Sphere(.5,'#B80C09')
```


    
![png](README_files/README_16_0.png)
    


## Difference:


```python
Box(1,'#0B4F6C') - Sphere(1.2)
```


    
![png](README_files/README_18_0.png)
    


## Intersection:


```python
Icosahedron(1,'#0B4F6C') & Sphere(1.1)
```


    
![png](README_files/README_20_0.png)
    


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


    
![png](README_files/README_23_0.png)
    


# Create videos!
Use the 't' (time) variable to control the animation


```python
x = Box(1,'palette(4*((x+sin(t))+(y+cos(t))**2+z**3))',palette = palette)
x += 'ry t'
x += 'rx t'

x.animate(frames = 60, framerate = 15, iterations = 100)
```

    Animating..: 100%|██████████| 59/59 [00:52<00:00,  1.12it/s]





    
![png](README_files/README_25_1.png)
    



# Transformations

## Translation


```python
Sphere(1) + 'dx .5'
```


    
![png](README_files/README_28_0.png)
    



```python
Sphere(1) + '(.1,.2,.3)'
```


    
![png](README_files/README_29_0.png)
    


## Scale


```python
Sphere(1) * .2
```


    
![png](README_files/README_31_0.png)
    



```python
Sphere(1) * (1,.2,1)
```


    
![png](README_files/README_32_0.png)
    


## Rotation


```python
Box(1) + 'rx pi/4'
```


    
![png](README_files/README_34_0.png)
    


## Advanced transformations

You can use x,y,z (and the time parameter, t) as variables in transformations such as translation, rotation, scale


```python
BoxFrame(1,.1,'#f44') + 'dx .2*y'
```


    
![png](README_files/README_37_0.png)
    


# Other operations

## Twist along an axis


```python
Box(1,'#f44').twist(2,'y')
```


    
![png](README_files/README_40_0.png)
    


## Create an "onion" shape


```python
# Create an onion from a box
x = Box(1,'#f44').onion()
# Cut a hole in the onion
x &= Shape('z')

x
```


    
![png](README_files/README_42_0.png)
    


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


    
![png](README_files/README_45_0.png)
    



    
![png](README_files/README_45_1.png)
    


## Smooth difference


```python
sphere = Sphere(1.1)
box = Box(1,'#f44')

# Normal difference
display(box - sphere)

# Smooth difference
display(box <<sd(.5)>> sphere)
```


    
![png](README_files/README_47_0.png)
    



    
![png](README_files/README_47_1.png)
    


## Smooth intersection


```python
sphere = Sphere(1)
box = Box(.9,'#f44')

# Normal intersection
display(box & sphere)

# Smooth intersection
display(box <<si(.5)>> sphere)
```


    
![png](README_files/README_49_0.png)
    



    
![png](README_files/README_49_1.png)
    



```python

```
