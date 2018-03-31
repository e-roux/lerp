
from numpy import (random, array, arange, linspace, float64,
				  pi, interp, sin, sort, unique)
from lerp import Mesh

random.seed(123)

x = linspace(0, 2 * pi, 10)
y = sort(unique(random.randint(0, 50000, size=20))).astype(float64)

data = random.random(x.size * y.size).reshape(x.size, y.size).astype(float64) * 10
m3d = Mesh(coords=[('x', x), ('y', y)],
		   data=data)

print(m3d)

print(m3d[3, 4])
xi = linspace(-1.5,  2 * pi + 1.5, 10)


res = m3d([x[3], x[4]], [y[4], y[7]])

print("*"*30)
print("* Résultat")
print("*"*30)

print(res)


# print("*"*30)
# for _x in m3d.rolling(x=1):
#     print(f"{_x[0].x.data:.2f}, {_x[1].data[0]:.2f}")
# print("*"*30)

# for _x in xi:
#     print(_x, m2d([_x]))
