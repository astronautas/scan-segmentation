package shape_modelling

import java.io.File

import junit.framework.{Assert, TestCase}
import org.junit.Test
import scalismo.io.{ActiveShapeModelIO, ImageIO, MeshIO}

class LikelyhoodCheckerTest extends TestCase {

  @Test
  def test(): Unit = {
    scalismo.initialize()

    val asm = ActiveShapeModelIO.readActiveShapeModel(new File("handedData/femur-asm.h5")).get
    val image = ImageIO.read3DScalarImage[Short](new File("handedData/test/30.nii")).get.map(_.toFloat)
    val mesh = MeshIO.readMesh(new File("handedData/test/30.stl")).get

    val parallel = LikelihoodChecker.likelihoodThatMeshFitsImage(asm, mesh, asm.preprocessor(image))
    val nonParallel = LikelihoodChecker.likelihoodThatMeshFitsImageNonParallel(asm, mesh, asm.preprocessor(image))

    Assert.assertEquals(parallel, nonParallel)
  }
}
