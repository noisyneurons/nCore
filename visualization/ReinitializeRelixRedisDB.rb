# '~/Code/Ruby/NN2012/nCore/visualization/ReinitializeRelixRedis.rb'
# This program deletes all of the redis database and recreates the "experimentNumber" single-number database

require_relative '../lib/core/SimulationDataStore'

lastExperimentNumber = $redis.get("experimentNumber")
logger.puts "\nLast Experiment Number=\t #{lastExperimentNumber}"

## --- DANGER ----###
$redis.flushdb

nextExperimentNumber = lastExperimentNumber.to_i + 1
ExperimentLogger.initializeExperimentNumber(nextExperimentNumber.to_s)
$redis.save

logger.puts "\nNext Experiment Number=\t #{$redis.get("experimentNumber")}"


