package rittenhouse.MAIN;

import rittenhouse.ARMS.LeftArm;
import rittenhouse.DROID.SentryDroid_B;
import rittenhouse.FACTORY.Sentry_AFactory;
import rittenhouse.FACTORY.Sentry_BFactory;
import rittenhouse.FACTORY.Sentry_CFactory;

public class RittenhouseDroids {
	
	public static void main(String[] args) {
		
		Sentry_AFactory A = new Sentry_AFactory("A-1");
		Sentry_BFactory B = new Sentry_BFactory("B-1");
		Sentry_CFactory C = new Sentry_CFactory("C-1");
		A.displayInfo();
		System.out.println();
		A.buildDroids_A(3).get("A1").displaySDA();
		System.out.println();
		SentryDroid_B b = B.buildDroids_B(30).get("B15");
		((LeftArm)(b.getArms().get("Left Arm"))).setOperational(false);
		b.displaySDB();
		
	}

}
