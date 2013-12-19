require_relative 'Utilities'


def ioFunction(aNetInput)
  h(aNetInput, 4)
end

def h(x, s)
  f(x) + ( -0.5 * (f(x + s) + f(x-s)) ) + 0.5
end

def f(x)
  1.0 / (1.0 + Math.exp(-1.0 * x))
end

puts ioFunction(-3.0)

puts ioFunction(-2.5)

puts ioFunction(-1.0)

puts ioFunction(0.0)

puts ioFunction(1.0)

puts ioFunction(2.5)

puts ioFunction(3.0)


def ioDerivativeFromNetInput(aNetInput)
  return j(aNetInput, 4.0)
end

def j(x, s)
  g(x, 0.0) - (0.5 * ( g(x,s) + g(x,(-1.0 * s))))
end

def g(x, s)
  Math.exp((-1.0 * x) + s)   /  ((Math.exp((-1.0 * x) + s))   + 1.0)  **  2.0
end

puts "\n\n"

puts ioDerivativeFromNetInput(-4.2)

puts ioDerivativeFromNetInput(-3.0)

puts ioDerivativeFromNetInput(-2.5)

puts ioDerivativeFromNetInput(-1.0)

puts ioDerivativeFromNetInput(0.0)

puts ioDerivativeFromNetInput(1.0)

puts ioDerivativeFromNetInput(2.5)

puts ioDerivativeFromNetInput(3.0)

puts ioDerivativeFromNetInput(4.2)