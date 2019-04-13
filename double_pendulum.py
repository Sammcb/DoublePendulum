from PIL import Image, ImageDraw
import numpy as np
from scipy.integrate import odeint
import random

DEBUG = False

def alpha(initial_cond, t, m1, m2, l1, l2, g):
  theta1, omega1, theta2, omega2 = initial_cond

  c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

  alpha1 = (m2*g*np.sin(theta2)*c-m2*s*(l1*omega1**2*c+l2*omega2**2)-(m1+m2)*g*np.sin(theta1))/l1/(m1+m2*s**2)
  alpha2 = ((m1+m2)*(l1*omega1**2*s-g*np.sin(theta2)+g*np.sin(theta1)*c)+m2*l2*omega2**2*s*c)/l2/(m1+m2*s**2)

  return omega1, alpha1, omega2, alpha2

# Copied from ColorsFinder project (https://github.com/Sammcb/ColorsFinder)
def calculate_colors(color1, color2, num_colors):
  delta = [color1[0] - color2[0], color1[1] - color2[1], color1[2] - color2[2]]

  colors = [color1 + (170,)]
  for i in range(1, num_colors + 1):
    colors.append((
      round(color1[0] - (i * (delta[0] / (num_colors + 1)))),
      round(color1[1] - (i * (delta[1] / (num_colors + 1)))),
      round(color1[2] - (i * (delta[2] / (num_colors + 1)))),
      170
    ))
  colors.append(color2 + (170,))

  return colors

# Generates art based on the movement of a double pendulum under randomized conditions
def generate_image(path):
  # Set up blank image
  size = (800, 600)
  img = Image.new('RGBA', size, color='white')
  d = ImageDraw.Draw(img)

  # Initial conditions
  m1 = random.uniform(1, 100)
  m2 = random.uniform(1, 100)
  l1 = random.uniform(50, 75)
  l2 = random.uniform(100, 150)
  g = random.uniform(1, 5)
  theta1 = random.uniform(0, 2*np.pi)
  theta2 = random.uniform(0, 2*np.pi)
  omega1 = random.uniform(3, 6)
  omega2 = random.uniform(0, 2)

  x0 = 400
  y0 = 300

  initial_cond = np.array([theta1, omega1, theta2, omega2])

  tmax = random.uniform(25, 100)
  dt = 0.001
  t = np.arange(0, tmax + dt, dt)

  # Print initial conditions
  if DEBUG:
    print('m1:     ' + str(m1))
    print('m2:     ' + str(m2))
    print('l1:     ' + str(l1))
    print('l2:     ' + str(l2))
    print('g:      ' + str(g))
    print('theta1: ' + str(theta1))
    print('theta2: ' + str(theta2))
    print('omega1: ' + str(omega1))
    print('omega2: ' + str(omega2))
    print('tmax:   ' + str(tmax))

  # Solve the coupled system of second order differential equations
  sol = odeint(alpha, initial_cond, t, args=(m1, m2, l1, l2, g))
  theta1, omega1, theta2, omega2 = sol[:,0], sol[:,1], sol[:,2], sol[:,3]
  x1 = x0 + l1 * np.sin(theta1)
  y1 = y0 + l1 * np.cos(theta1)
  x2 = x1 + l2 * np.sin(theta2)
  y2 = y1 + l2 * np.cos(theta2)

  omega2_max = max(np.abs(omega2))
  omega2_min = min(np.abs(omega2))

  color1 = (0, 0, 255)
  color2 = (252, 1, 7)
  num_colors = int(omega2_max*10) - int(omega2_min*10)

  colors = calculate_colors(color1, color2, num_colors)

  # Draw the points on the image
  for i in range(0, t.size):
    if img.load()[round(x2[i]), round(y2[i])] == (255, 255, 255, 255):
      d.point((round(x2[i]), round(y2[i])), fill=colors[int(abs(omega2[i])*10) - int(omega2_min*10)])
    else:
      # Calculate new color with alpha using Painter's algorithm
      old_c = img.load()[round(x2[i]), round(y2[i])]
      add_c = colors[int(abs(omega2[i])*10) - int(omega2_min*10)]

      old_a = old_c[3] / 255
      add_a = add_c[3] / 255
      new_a = old_a + add_a * (1 - old_a)

      new_c = [round(c3/new_a) for c3 in [sum(c) for c in zip([old_a*c1 for c1 in old_c], [add_a*(1-old_a)*c2 for c2 in add_c])]]

      d.point((round(x2[i]), round(y2[i])), fill=(new_c[0], new_c[1], new_c[2], int(new_a * 255)))

  img.save(path)

def main():
  # Set to true to print initial conditions to console
  DEBUG = False

  # Path to save the image at
  path = 'test.png'
  generate_image(path)

main()
