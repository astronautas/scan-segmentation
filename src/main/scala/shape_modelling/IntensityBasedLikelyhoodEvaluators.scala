package shape_modelling

import breeze.linalg.DenseVector
import scalismo.geometry.{Point3D, SquareMatrix, Vector3D, _3D}
import scalismo.registration.RigidTransformationSpace
import scalismo.sampling.DistributionEvaluator
import scalismo.statisticalmodel.NDimensionalNormalDistribution
import scalismo.statisticalmodel.asm.{ActiveShapeModel, PreprocessedImage}
import shape_modelling.MCMC.ShapeParameters

object IntensityBasedLikelyhoodEvaluators {
  case class IntensityBasedLikeliHoodEvaluatorForShapeFitting(asm: ActiveShapeModel, preprocessedImage: PreprocessedImage,
                                                                 sdev: Double = 0.5) extends DistributionEvaluator[ShapeParameters] {

    val uncertainty = NDimensionalNormalDistribution(Vector3D(0f, 0f, 0f), SquareMatrix.eye[_3D] * (sdev * sdev))

    override def logValue(theta: ShapeParameters): Double = {
      val value = LikelihoodChecker.likelihoodThatMeshFitsImage(asm, asm.statisticalModel.instance(theta.modelCoefficients), preprocessedImage)
      value
    }
  }

  case class IntensityBasedLikeliHoodEvaluatorForRigidFitting(asm: ActiveShapeModel, preprocessedImage: PreprocessedImage,
                                                              sdev: Double = 0.5) extends DistributionEvaluator[ShapeParameters] {

    val uncertainty = NDimensionalNormalDistribution(Vector3D(0f, 0f, 0f), SquareMatrix.eye[_3D] * (sdev * sdev))

    override def logValue(theta: ShapeParameters): Double = {
      // Need to consider rot/trans as well, so we first apply the transform to the mesh here.
      var mesh = asm.statisticalModel.instance(theta.modelCoefficients)
      val current_translation_coeffs = Segmentation.center_of_mass + theta.translationParameters
      val current_CoM: Point3D = new Point3D(current_translation_coeffs.valueAt(0), current_translation_coeffs.valueAt(1), current_translation_coeffs.valueAt(2))

      val rigidTransSpace = RigidTransformationSpace[_3D](current_CoM)
      val rigidtrans = rigidTransSpace.transformForParameters(DenseVector.vertcat(theta.translationParameters, theta.rotationParameters))

      mesh = mesh.transform(rigidtrans)

      val value = LikelihoodChecker.likelihoodThatMeshFitsImage(asm, mesh, preprocessedImage)
      value
    }
  }

  case class IntensityBasedLikeliHoodEvaluatorForRigidFittingFast(asm: ActiveShapeModel, preprocessedImage: PreprocessedImage,
                                                              sdev: Double = 0.5) extends DistributionEvaluator[ShapeParameters] {

    val uncertainty = NDimensionalNormalDistribution(Vector3D(0f, 0f, 0f), SquareMatrix.eye[_3D] * (sdev * sdev))

    override def logValue(theta: ShapeParameters): Double = {
      // Need to consider rot/trans as well, so we first apply the transform to the mesh here.
      var mesh = asm.statisticalModel.instance(theta.modelCoefficients)
      val current_translation_coeffs = Segmentation.center_of_mass + theta.translationParameters
      val current_CoM: Point3D = new Point3D(current_translation_coeffs.valueAt(0), current_translation_coeffs.valueAt(1), current_translation_coeffs.valueAt(2))

      val rigidTransSpace = RigidTransformationSpace[_3D](current_CoM)
      val rigidtrans = rigidTransSpace.transformForParameters(DenseVector.vertcat(theta.translationParameters, theta.rotationParameters))

      mesh = mesh.transform(rigidtrans)

      val value = LikelihoodChecker.likelihoodThatMeshFitsImageFast(asm, mesh, preprocessedImage)
      value
    }
  }
}
