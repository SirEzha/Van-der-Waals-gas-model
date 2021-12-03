import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.distance import pdist, squareform

# 2d variant for simplicity

class gas_simulation_2d:

    def __init__(self, particles_count = 2500, mass = 5e-20, effective_radius = 1.5e-10, area = 1e-14, T = 300):
        self.partcount = particles_count
        self.mass = mass
        self.radius = effective_radius
        self.area = area
        self.side_length = (area ** (1/2))
        self.temp = T
        self.vel_normalized = (3 * 1.87e-23 * self.temp / self.mass) ** (1/2)
        self.set_particles()
        self.set_graphs()

    def set_particles(self):
        # set positions and velocities with array of type [[x1, y1], [x2, y2], ..., [xn, yn]]
        dists = np.linspace(self.radius * 5, self.side_length - self.radius * 5, round(self.partcount ** (1/2)))
        self.pos = np.zeros((self.partcount, 2))
        for i in range(round(self.partcount ** (1/2))):
            for j in range(round(self.partcount ** (1/2))):
                self.pos[i * round(self.partcount ** (1/2)) + j, :] = np.array([dists[i] + random.rand(1)[0] * (self.radius * 0.2), dists[j] + random.rand(1)[0] * (self.radius * 0.2)])
        self.vel = random.uniform(-1, 1, (self.partcount, 2)) * self.vel_normalized
        self.vel_hist_data = np.zeros(self.partcount)
        for i in range(self.partcount):
            self.vel_hist_data[i] = (self.vel[i, 0] ** 2 + self.vel[i, 1] ** 2) ** (1/2)

    def set_graphs(self):
        self.figure = plt.figure()
        self.ax1 = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=2, xlim=(-self.radius * 5, self.side_length + self.radius * 5), ylim=(-self.radius * 5, self.side_length + self.radius * 5))
        self.ax2 = plt.subplot2grid((3, 3), (0, 2), xlim=(0, 2))

    def step(self, dt):

        # updating positions of the particles
        self.pos = self.pos + self.vel * dt

        # finding colliding particles, can be improved by creating a grid
        distances = squareform(pdist(self.pos))
        ind1, ind2 = np.where(distances < 2 * self.radius)
        unique = (ind1 < ind2)
        ind1 = ind1[unique]
        ind2 = ind2[unique]

        # calculating the result of the collision
        for i1, i2 in zip(ind1, ind2):
            vel_cm = (self.vel[i1] + self.vel[i2]) / 2
            vec_normal = self.pos[i1] - self.pos[i2]
            vel_normal = self.vel[i1] - self.vel[i2]
            vel_change = 2 * np.dot(vec_normal, vel_normal) * vec_normal / np.dot(vec_normal, vec_normal) - vel_normal
            self.vel[i1] = vel_cm - vel_change / 2
            self.vel[i2] = vel_cm + vel_change / 2
            self.pos[i1] += self.vel[i1] * dt
            self.pos[i2] += self.vel[i2] * dt

        # finding particles colliding with the wall
        for i in range(self.partcount):
            if self.pos[i, [0]] - self.radius < 0:
                self.vel[i, [0]] = -self.vel[i, [0]]
                self.pos[i, [0]] = self.radius * 1.01
            if self.pos[i, [0]] + self.radius > self.side_length:
                self.vel[i, [0]] = -self.vel[i, [0]]
                self.pos[i, [0]] = self.side_length - (self.radius * 1.01)
            if self.pos[i, [1]] - self.radius < 0:
                self.vel[i, [1]] = -self.vel[i, [1]]
                self.pos[i, [1]] = self.radius * 1.01
            if self.pos[i, [1]] + self.radius > self.side_length:
                self.vel[i, [1]] = -self.vel[i, [1]]
                self.pos[i, [1]] = self.side_length - (self.radius * 1.01)

def init():
    global abobus_2d
    molecules.set_data([], [])
    return molecules,

def animate(a):
    global abobus_2d, dt, bars, data
    abobus_2d.step(dt)
    molecules.set_data(abobus_2d.pos[:, 0], abobus_2d.pos[:, 1])
    #molecules.set_color(random.choice(['b', 'g', 'r', 'c', 'm', 'y']))
    ms = int(fig.dpi * 2 * abobus_2d.radius * fig.get_figwidth() / np.diff(ax1.get_xbound())[0])
    molecules.set_markersize(1)
    for i in range(abobus_2d.partcount):
        data[i] = (abobus_2d.vel[i, 0] ** 2 + abobus_2d.vel[i, 1] ** 2) ** (1/2)
    _, _, bars = ax2.hist(data, bins = 60, lw=1, color='b')
    #plt.savefig(str(a) + ".png")
    return bars.patches + [molecules]

def my_dist(v, mass, temp):
    return (mass / (1.87e-23 * temp)) * v * np.exp(-mass * (v ** 2) / (2 * 1.87e-23 * temp)) * 75

x = np.linspace(0, 5, 100)
p = my_dist(x, 5e-20, 300)
abobus_2d = gas_simulation_2d()
dt = 1. / 300000000
fig = abobus_2d.figure
ax1 = abobus_2d.ax1
ax2 = abobus_2d.ax2
ax2.plot(x, p)
data = abobus_2d.vel_hist_data
_, _, bars = ax2.hist([], bins = 60, lw=1)
molecules, = ax1.plot([], [], 'p')
ani = animation.FuncAnimation(fig, animate, frames=100, interval=10, blit=True, init_func=init)
#ani.save('particle_box_disco2.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
plt.show()