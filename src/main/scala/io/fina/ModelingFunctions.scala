package io.fina
import com.redis.RedisClient
import io.evvo.agent.{CreatorFunction, CrossoverFunction, DeletorFunction, MutatorFunction}
import io.evvo.builtin.deletors.DeleteDominated
import io.evvo.island.population.{Maximize, Minimize, Objective, Scored}
import io.evvo.island._
import io.evvo.migration.redis.{RedisEmigrator, RedisImmigrator, RedisParetoFrontierRecorder}
import io.fina.fina.IOMap

import scala.concurrent.duration._

package object fina {
  type IOMap = Map[Double, Double]
}

object iomap {

  /** Represents an input/output table where the input is evaluated by some function to produce the
    * output. Mapped double -> double to allow for smaller increments in inputs as well as decimal
    * values in outputs.
    */
  /** A creator that generates `IOMap`s by reusing the given input values and generating new output
    * values using a normal distribution centered around the given output data.
    */
  case class IOMapGenerator(targetMap: Map[Double, Double])
      extends CreatorFunction[IOMap]("IOMapGenerator") {
    override def create(): Iterable[IOMap] = {
      val mean = targetMap.values.sum / targetMap.size
      val stdDev = Math.sqrt(
        targetMap.values
          .map(_ - mean)
          .map(t => t * t)
          .sum / targetMap.size)

      Vector.fill(32)(
        Map.from(
          targetMap.keys
            .map(k => (k, util.Random.nextGaussian() * stdDev + mean))
        ))
    }
  }

  case class MinimizeLoss(targetMap: Map[Double, Double])
      extends Objective[IOMap]("MinimizeLoss", Minimize()) {
    override protected def objective(solution: IOMap): Double = {
      targetMap.keys
        .map(k => Math.abs(targetMap(k) - solution(k)))
        .sum
    }
  }

  private def selectRandomKeys(iomap: IOMap): (Double, Double) = {
    var key1 = 0.0
    var key2 = 0.0

    while (key1 == key2) {
      key1 = iomap.keys.toList(util.Random.nextInt(iomap.keys.size))
      key2 = iomap.keys.toList(util.Random.nextInt(iomap.keys.size))
    }

    (key1, key2)
  }

  case class OutputSwapper() extends MutatorFunction[IOMap]("OutputSwapper") {
    override def mutate(iomap: IOMap): IOMap = {
      val keys = selectRandomKeys(iomap)

      iomap
        .updated(keys._1, iomap(keys._2))
        .updated(keys._2, iomap(keys._1))
    }
  }

  case class CrossoverModifier() extends CrossoverFunction[IOMap]("Crossover") {
    override protected def crossover(iomap1: IOMap, iomap2: IOMap): IOMap = {
      val keys = selectRandomKeys(iomap1)
      val key1 = keys._1
      val key2 = keys._2

      iomap1.updated(key1, iomap2(key2))
    }
  }
}

object ModelingFunctions {
  def main(args: Array[String]): Unit = {
    val redisClient = new RedisClient("localhost", 6379)
    val duration = "10".toInt.seconds

    val targetMap: Map[Double, Double] = Map.from(
      Range(0, 10).map(i => (i.toDouble, i.toDouble * i))
    )

    val evvoIsland = new EvvoIsland[IOMap](
      creators = Vector(iomap.IOMapGenerator(targetMap)),
      mutators = Vector(iomap.OutputSwapper(), iomap.CrossoverModifier()),
      deletors = Vector(DeleteDominated()),
      fitnesses = Vector(iomap.MinimizeLoss(targetMap)),
      immigrator = new RedisImmigrator[IOMap](redisClient),
      immigrationStrategy = AllowAllImmigrationStrategy(),
      emigrator = new RedisEmigrator[IOMap](redisClient),
      emigrationStrategy = RandomSampleEmigrationStrategy(32),
      loggingStrategy = LogPopulationLoggingStrategy(),
      paretoFrontierRecorder = new RedisParetoFrontierRecorder[IOMap](redisClient)
    )

    evvoIsland.runBlocking(StopAfter(duration))
    println(evvoIsland.currentParetoFrontier().toCsv())
    sys.exit(0)
  }
}
