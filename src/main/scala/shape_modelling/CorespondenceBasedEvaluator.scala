package shape_modelling

import breeze.linalg.DenseVector
import scalismo.common.PointId
import scalismo.geometry._
import scalismo.registration.RigidTransformationSpace
import scalismo.sampling.DistributionEvaluator
import scalismo.statisticalmodel.{NDimensionalNormalDistribution, StatisticalMeshModel}
import shape_modelling.MCMC.ShapeParameters

case class CorespondenceBasedEvaluator(model: StatisticalMeshModel, correspondences : Seq[(PointId, Point[_3D])],
                                       tolerance: Double) extends DistributionEvaluator[ShapeParameters] {

  val uncertainty = NDimensionalNormalDistribution(Vector3D(0f,0f,0f), SquareMatrix.eye[_3D]* tolerance)

  override def logValue(theta: ShapeParameters): Double = {
    var mesh = model.instance(theta.modelCoefficients)
    val current_translation_coeffs = Segmentation.center_of_mass + theta.translationParameters
    val current_CoM: Point3D = new Point3D(current_translation_coeffs.valueAt(0), current_translation_coeffs.valueAt(1), current_translation_coeffs.valueAt(2))

    val rigidTransSpace = RigidTransformationSpace[_3D](current_CoM)
    val rigidtrans = rigidTransSpace.transformForParameters(DenseVector.vertcat(theta.translationParameters, theta.rotationParameters))

    var currModelInstance = mesh.transform(rigidtrans)

    val likelihoods = correspondences.map { case (id, targetPoint) =>
      val modelInstancePoint = currModelInstance.point(id)
      val observedDeformation =  targetPoint - modelInstancePoint

      uncertainty.logpdf(observedDeformation)
    }

    val loglikelihood = likelihoods.sum
    loglikelihood
  }
}
