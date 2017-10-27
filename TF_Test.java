import org.tensorflow.*;
import java.util.Arrays;
import java.nio.file.*;
import java.nio.*;
import java.util.*;
import java.io.FileInputStream;
import java.nio.ByteBuffer;

public class TF_Test {
  public static void main(String[] args) throws Exception {
    System.out.println(TensorFlow.version());
    try{
	MnistReader mrTest = new MnistReader("./data/MNIST/t10k-labels.idx1-ubyte", 
					     "./data/MNIST/t10k-images.idx3-ubyte");		
	
	//restores the saved model from the /model directory
	SavedModelBundle smb = SavedModelBundle.load("./model", "serve");
	//starts a TF session
	Session s = smb.session();

	int correct_prediction = 0;
	for(int i = 0; i<mrTest.size(); i++) {

		//create float buffer for the image
		FloatBuffer fb = FloatBuffer.allocate(784);
	
		//read image bytes into float buffer
		byte[] imgData = mrTest.readNextImage();
		for(byte b : imgData) {
			fb.put((float)(b & 0xFF)/255.0f);
		}
		fb.rewind();

		float[] keep_prob_arr = new float[1024];
		Arrays.fill(keep_prob_arr, 1f);

	
		Tensor inputTensor = Tensor.create(new long [] {784}, fb);
		Tensor keep_prob = Tensor.create(new long[] {1,1024}, FloatBuffer.wrap(keep_prob_arr));
	
		Tensor result = s.runner().feed("x", inputTensor)
					  .feed("y_true", keep_prob)
					  .fetch("output_tensor")
					  .run().get(0);

	
		long[] r = new long[1];
		long[] res = result.copyTo(r);
		if(res[0] == mrTest.readNextLabel())
			correct_prediction++;
	}
	System.out.println("Out of "+mrTest.size()+" Examples, "+correct_prediction+" were correct.");
    }catch(Exception e){
	System.out.println("Exception ---------------------------------------- Exception");
	e.printStackTrace();
    }
  }
}
