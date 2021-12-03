import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.distance import pdist, squareform

class gas_simulation_3d:

    def __init__(self, particles_count = 2197, mass = 5e-20, effective_radius = 2e-10, volume = 1e-23, T = 300):
        self.partcount = particles_count
        self.mass = mass
        self.radius = effective_radius
        self.volume = volume
        self.side_length = (volume ** (1/3))
        self.temp = T
        self.vel_normalized = (3 * 1.87e-23 * self.temp / self.mass) ** (1/2)
        self.pressure = 0
        self.dp = 0
        self.counter = 0
        self.left_hand_side = 0
        self.mass_non_normed = 5
        self.b = self.partcount * ((4/3) * np.pi * (self.radius ** 3))
        self.set_particles()
        self.set_graphs()

    def set_particles(self):
        # set positions and velocities with array of type [[x1, y1, z1], [x2, y2, z2], ..., [xn, yn, zn]]
        dists = np.linspace(self.radius * 5, self.side_length - self.radius * 5, round(self.partcount ** (1 / 3)))
        self.pos = np.zeros((self.partcount, 3))
        for i in range(round(self.partcount ** (1 / 3))):
            for j in range(round(self.partcount ** (1 / 3))):
                for k in range(round(self.partcount ** (1 / 3))):
                    self.pos[i * round(self.partcount ** (2 / 3)) + j * round(self.partcount ** (1 / 3)) + k, :] = np.array(
                        [dists[i] + random.rand(1)[0] * (self.radius * 0.2),
                         dists[j] + random.rand(1)[0] * (self.radius * 0.2), dists[k] + random.rand(1)[0] * (self.radius * 0.2)])
        self.vel = random.uniform(-1, 1, (self.partcount, 3)) * self.vel_normalized
        self.vel_hist_data = np.zeros(self.partcount)
        for i in range(self.partcount):
            self.vel_hist_data[i] = (self.vel[i, 0] ** 2 + self.vel[i, 1] ** 2 + self.vel[i, 2] ** 2) ** (1 / 2)

    def set_graphs(self):
        self.figure = plt.figure(figsize=(11, 8))
        self.ax1 = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=2, projection='3d')
        box_limits = np.array([0, self.side_length])
        self.ax1.set_xlim3d(box_limits)
        self.ax1.set_ylim3d(box_limits)
        self.ax1.set_zlim3d(box_limits)
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_zlabel('Z')
        self.ax2 = plt.subplot2grid((3, 3), (0, 2))
        self.ax2.set_xlabel('Speed')
        self.ax2.set_ylabel('Frequency')


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

        # finding particles colliding with the wall and the change of momentum
        for i in range(self.partcount):
            if self.pos[i, [0]] - self.radius < 0:
                self.vel[i, [0]] = -self.vel[i, [0]]
                self.pos[i, [0]] = self.radius * 1.01

                # finding change of momentum
                self.dp += 2 * abs(self.vel[i, [0]]) * self.mass_non_normed
                self.counter += 1
            if self.pos[i, [0]] + self.radius > self.side_length:
                self.vel[i, [0]] = -self.vel[i, [0]]
                self.pos[i, [0]] = self.side_length - (self.radius * 1.01)
            if self.pos[i, [1]] - self.radius < 0:
                self.vel[i, [1]] = -self.vel[i, [1]]
                self.pos[i, [1]] = self.radius * 1.01
            if self.pos[i, [1]] + self.radius > self.side_length:
                self.vel[i, [1]] = -self.vel[i, [1]]
                self.pos[i, [1]] = self.side_length - (self.radius * 1.01)
            if self.pos[i, [2]] - self.radius < 0:
                self.vel[i, [2]] = -self.vel[i, [2]]
                self.pos[i, [2]] = self.radius * 1.01
            if self.pos[i, [2]] + self.radius > self.side_length:
                self.vel[i, [2]] = -self.vel[i, [2]]
                self.pos[i, [2]] = self.side_length - (self.radius * 1.01)

        # finding the pressure
        # self.pressure = self.dp / (self.counter * dt * (self.side_length ** 2))
        # print(self.pressure)
        # self.left_hand_side = self.pressure * (self.side_length ** 3 - self.b)
        # print(self.left_hand_side)
        # print(self.counter)
        # if self.counter > 3e3:
        #     self.dp = 0
        #     self.side_length *= 1.2
        #     self.counter = 0



def init():
    global abobus_3d
    molecules.set_data_3d([], [], [])
    ax2.set_xlim(0, 1.5)
    ax2.set_ylim(0, 2)
    return molecules,

def animate(a):
    global abobus_3d, dt, bars, data
    ax2.clear()
    ax2.set_xlim(0, 1.5)
    ax2.set_ylim(0, 2)
    abobus_3d.step(dt)
    molecules.set_data_3d(abobus_3d.pos[:, 0], abobus_3d.pos[:, 1], abobus_3d.pos[:, 2])
    molecules.set_markersize(1)
    for i in range(abobus_3d.partcount):
        data[i] = (abobus_3d.vel[i, 0] ** 2 + abobus_3d.vel[i, 1] ** 2 + abobus_3d.vel[i, 2] ** 2) ** (1 / 2)
    _, _, bars = ax2.hist(data, bins = 100, lw=1, density=True, alpha=0.75)
    ax2.plot(x, p)
    molecules.set_color(random.choice(['b', 'g', 'r', 'c', 'm', 'y']))
    plt.savefig(str(a) + ".png")
    return bars.patches + [molecules]

def my_dist(v, mass, temp):
    return ((2 / np.pi) ** (1/2)) * ((mass / (1.87e-23 * temp)) ** (3/2)) * (v ** 2) * np.exp(-mass * (v ** 2) / (2 * 1.87e-23 * temp))


abobus_3d = gas_simulation_3d()
x = np.linspace(0, 5, 100)
p = my_dist(x, abobus_3d.mass, 300)
dt = 1. / 500000000
fig = abobus_3d.figure
ax1 = abobus_3d.ax1
ax2 = abobus_3d.ax2
ax2.plot(x, p)
data = abobus_3d.vel_hist_data
_, _, bars = ax2.hist([], bins = 100, lw=1, density=True, alpha=0.75)
molecules, = ax1.plot([], [], [], 'p')
ani = animation.FuncAnimation(fig, animate, frames=300, interval=40, blit=True, init_func=init)
#ani.save('particle_box_3d_test.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
plt.show()