package shape_modelling

import scalismo.common.PointId
import scalismo.geometry.{SquareMatrix, Vector3D, _3D, Point}
import scalismo.sampling.DistributionEvaluator
import scalismo.statisticalmodel.{NDimensionalNormalDistribution, StatisticalMeshModel}
import shape_modelling.MCMC.ShapeParameters

case class CorespondenceBasedEvaluator(model: StatisticalMeshModel, correspondences : Seq[(PointId, Point[_3D])],
                                       tolerance: Double) extends DistributionEvaluator[ShapeParameters] {

  val uncertainty = NDimensionalNormalDistribution(Vector3D(0f,0f,0f), SquareMatrix.eye[_3D]* tolerance)

  override def logValue(theta: ShapeParameters): Double = {

    val currModelInstance = model.instance(theta.modelCoefficients)

    val likelihoods = correspondences.map { case (id, targetPoint) =>
      val modelInstancePoint =  currModelInstance.point(id)
      val observedDeformation =  targetPoint - modelInstancePoint

      uncertainty.logpdf(observedDeformation)
    }

    val loglikelihood = likelihoods.sum
    loglikelihood
  }
}
