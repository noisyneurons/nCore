require_relative 'Utilities'
require_relative 'DataSet'


anorm = NormalDistribution.new(0.0,1.0)

100.times do
puts anorm.get_rng
end


#learningLayers = [[2]]
#allNeuronLayers = [[0], [1], [2], [3]]
#
#learningLayers = [[3]]
#allNeuronLayers = [[], [], [], [3]]
#
#puts allNeuronLayers.flatten.empty?
#
##propagatingLayers, controllingLayers = layerDetermination(learningLayers, allNeuronLayers)
##
##puts "propagatingLayers = #{propagatingLayers}\n\n"
##
##puts "controllingLayers = #{controllingLayers}"