package org.reroutlab.code.auav.routines;

import java.util.HashMap;
import org.eclipse.californium.core.CoapHandler;
import org.eclipse.californium.core.CoapResponse;

//Tensorflow Test Imports:
import org.tensorflow.*;
import java.util.Arrays;
import java.nio.file.*;
import java.util.*;
import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.io.File;
/**
 * BasicRoutine takes off, calibrates camera and lands
 * It does not sense from it's environment
 * Invoke this routine through external commands driver
 * <br><b>URL: coap:\\localhost:port\cr?</b>
 * <br>Example: coap:\\127.0.0.1:5117\cr?dn=rtn-dc=start-dp=BasicRoutine
 * <br>
 * Note: all AUAV drivers have resource component "cr" for coap resource
 *
 * @author  Christopher Charles Stewart
 * @version 1.0.3
 * @since   2017-10-01
 */
public class BasicRoutine extends org.reroutlab.code.auav.routines.AuavRoutines{
		/**
		 *	 Check forceStop often and safely end routine if set
		 */
		public boolean forceStop = false;

		/**
		 *	 Routines are Java Threads.  The run() function is the 
		 *	 starting point for execution. 
		 * @version 1.0.1
		 * @since   2017-10-01			 
		 */
		public void run() {	
			
				//this loads the JNI for tensorflow java on linux (assumes jar is in $AUAVHome/routines
				File JNI = new File("../external/jni/libtensorflow_jni.so");
				System.load(JNI.getAbsolutePath());
				System.out.println(TensorFlow.version());
				//loads model directory into a routine (assumes jar is in $AUAVHome/routines)
				File Model = new File("../Models/MNISTmodel/");
				SavedModelBundle smb = SavedModelBundle.load(Model.getAbsolutePath(), "serve");
				Session s = smb.session();

				String succ = "";
				succ = invokeDriver("org.reroutlab.code.auav.drivers.FlyDroneDriver",
						     "dc=lft-dp=AUAVsim", chResp );
				rtnSpin();
				rtnLock("free");

				succ = invokeDriver("org.reroutlab.code.auav.drivers.DroneGimbalDriver",
						    "dc=cal-dp=AUAVsim", chResp );
				rtnSpin();
				System.out.println("BasicRoutine: " + resp);
				rtnLock("free");
				succ = invokeDriver("org.reroutlab.code.auav.drivers.FlyDroneDriver",
						    "dc=lnd-dp=AUAVsim", chResp );
				rtnSpin();
				System.out.println("BasicRoutine: " + resp);
		}
				

		//  The code below is mostly template material
		//  Most routines will not change the code below
		//
		//
		//
		//
		//
		//  Christopher Stewart
		//  2017-10-1
		//

		private Thread t = null;
		private String csLock = "free";
		private String resp="";
		CoapHandler chResp = new CoapHandler() {
						@Override public void onLoad(CoapResponse response) {
								resp = response.getResponseText();
								rtnLock("barrier-1");
						}
						
						@Override public void onError() {
								System.err.println("FAILED");
								rtnLock("barrier-1");
						}};
				


		public BasicRoutine() {t = new Thread (this, "Main Thread");	}
		public String startRoutine() {
				if (t != null) {
						t.start(); return "BasicRoutine: Started";
				}
				return "BasicRoutine not Initialized";
		}
		public String stopRoutine() {
				forceStop = true;	return "BasicRoutine: Force Stop set";
		}
		synchronized void rtnLock(String value) {
				csLock = value;
		}
		public void rtnSpin() {
				while (csLock.equals("barrier-1") == false) {
						try { Thread.sleep(1000); }
						catch (Exception e) {}
				}

		}
		

		
}
