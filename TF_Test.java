import org.tensorflow.*;
import java.util.Arrays;
import java.nio.file.*;
import java.nio.*;
import java.util.*;

public class TF_Test {
  public static void main(String[] args) throws Exception {
    System.out.println(TensorFlow.version());
    try{
	SavedModelBundle smb = SavedModelBundle.load("./model", "serve");
	Session s = smb.session();
    }catch(Exception e){

    }
  }
}
