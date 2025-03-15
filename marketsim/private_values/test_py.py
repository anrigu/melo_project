
from marketsim.private_values.private_values import PrivateValues



p = PrivateValues(q_max=3, val_var=1e6)

print(p.values)

# p.value_at_position(-6)


# sum(p.values[0:p.offset]) + p.values[0]*3


print(p.value_at_position(3))

print(sum(p.values[p.offset:]))

print(p.value_at_position(-3))

print(p.value_at_position(3))


