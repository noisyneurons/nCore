require_relative 'Utilities'


#self.error = ioDerivativeFromNetInput(netInput) * ((netInput - target) / distanceBetweenSelfOrgTargets )


#   targetMinus                                       targetPlus
#        x                        0           e            x
#       -1                        0                       +1


#     (2.0 * ( (netInput - targetMinus) / (targetPlus - targetMinus)))  -  1.0


ioDerivativeFromNetInput = 1.0
netInput = -1.25
targetPlus = 2.5
targetMinus = -1.0 * targetPlus
distanceBetweenTargets = targetPlus - targetMinus


error = ioDerivativeFromNetInput * (2.0 * (  (netInput - targetMinus)/distanceBetweenTargets  )  -  1.0)


puts "error = #{error}"