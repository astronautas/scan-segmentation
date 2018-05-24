package shape_modelling

import java.io.File

import breeze.linalg.DenseVector
import junit.framework.{Assert, TestCase}
import org.junit.Test
import scalismo.io.{ActiveShapeModelIO, ImageIO, MeshIO}
import shape_modelling.MCMC.ShapeParameters

class LikelyhoodCheckerTest extends TestCase {

  @Test
  def test(): Unit = {
    scalismo.initialize()

    val asm = ActiveShapeModelIO.readActiveShapeModel(new File("handedDataNonAligned/femur-asm.h5")).get
    val image = ImageIO.read3DScalarImage[Short](new File("handedDataNonAligned/test/30.nii")).get.map(_.toFloat)
    val mesh = MeshIO.readMesh(new File("handedDataNonAligned/test/30.stl")).get

    var coeffs = ShapeParameters(DenseVector.zeros[Float](3), DenseVector.zeros[Float](3), asm.statisticalModel.coefficients(mesh))
    val parallel = LikelihoodChecker.likelihoodThatMeshFitsImage(asm, mesh, asm.preprocessor(image)) + MCMC.ShapePriorEvaluator(asm.statisticalModel).logValue(coeffs)
    val nonParallel = LikelihoodChecker.likelihoodThatMeshFitsImageNonParallel(asm, mesh, asm.preprocessor(image)) + MCMC.ShapePriorEvaluator(asm.statisticalModel).logValue(coeffs)

    Assert.assertEquals(parallel, nonParallel)
  }
}
