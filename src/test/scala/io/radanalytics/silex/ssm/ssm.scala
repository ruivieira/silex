/*
 * ssm.scala
 * author:  Rui Vieira <rui@redhat.com>
 *
 * Copyright (c) 2018 Red Hat, Inc.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.radanalytics.silex.ssm

import breeze.linalg.{DenseMatrix, DenseVector}
import org.scalatest._

class SsmStructureSpec extends FlatSpec with Matchers {
  it should "create a locally constant structure with the correct dimensions" in {
    val structure = Structure.locallyConstant(1.0)
    assert(structure.F.cols.equals(1))
    assert(structure.F.rows.equals(1))
    assert(structure.G.cols.equals(1))
    assert(structure.G.rows.equals(1))
    assert(structure.W.cols.equals(1))
    assert(structure.W.rows.equals(1))
  }
  it should "create a locally linear structure with the correct dimensions" in {
    val structure = Structure.locallyLinear(DenseMatrix.eye[Double](2))
    assert(structure.F.cols.equals(1))
    assert(structure.F.rows.equals(2))
    assert(structure.G.cols.equals(2))
    assert(structure.G.rows.equals(2))
    assert(structure.W.cols.equals(2))
    assert(structure.W.rows.equals(2))
  }
  it should "create a fourier structure (1 harmonic) with the correct dimensions" in {
    val structure = Structure.cyclicFourier(period = 100,
      harmonics = 1, W = DenseMatrix.eye[Double](2))
    assert(structure.F.cols.equals(1))
    assert(structure.F.rows.equals(2))
    assert(structure.G.cols.equals(2))
    assert(structure.G.rows.equals(2))
    assert(structure.W.cols.equals(2))
    assert(structure.W.rows.equals(2))
  }
  it should "create a fourier structure (2 harmonics) with the correct dimensions" in {
    val structure = Structure.cyclicFourier(period = 100,
      harmonics = 2, W = DenseMatrix.eye[Double](4))
    assert(structure.F.cols.equals(1))
    assert(structure.F.rows.equals(4))
    assert(structure.G.cols.equals(4))
    assert(structure.G.rows.equals(4))
    assert(structure.W.cols.equals(4))
    assert(structure.W.rows.equals(4))
  }
  it should "create a fourier structure (3 harmonics) with the correct dimensions" in {
    val structure = Structure.cyclicFourier(period = 100,
      harmonics = 3, W = DenseMatrix.eye[Double](6))
    assert(structure.F.cols.equals(1))
    assert(structure.F.rows.equals(6))
    assert(structure.G.cols.equals(6))
    assert(structure.G.rows.equals(6))
    assert(structure.W.cols.equals(6))
    assert(structure.W.rows.equals(6))
  }

}

class GaussianDlmSpec extends FlatSpec with Matchers {
  it should "keep state dimensions on propagation with locally constant structure" in {
    val structure = Structure.locallyConstant(1.0)
    val gaussianDLM = new GaussianDLM(structure = structure, V = 1.0)
    val prior = DenseVector.zeros[Double](1)
    val nextState = gaussianDLM.nextState(prior)

    assert(nextState.length.equals(1))
  }
}