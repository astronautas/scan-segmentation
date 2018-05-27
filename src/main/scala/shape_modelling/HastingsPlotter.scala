package shape_modelling

import scalax.chart.api._

import scala.collection.mutable.ListBuffer

class HastingsPlotter(frequency: Int) {
  var probabilities: ListBuffer[(Double, Double)] = ListBuffer()
  var lastFreq: Int = frequency

  val series = new XYSeries("Prob/Iteration")
  val chart = XYLineChart(series)
  chart.show()

  def offer(iteration: Double, value: Double) = {
    val tupl = (iteration, value)
    probabilities += tupl

    lastFreq -= 1

    if (lastFreq == 0) {
      swing.Swing onEDT {
        series.add(iteration, value)
      }

      lastFreq = frequency
    }
  }
}
