package rittenhouse.FACTORY;

import java.util.TreeMap;

import rittenhouse.DROID.SentryDroid_C;

public class Sentry_CFactory extends DroidFactory {
	
	public static int BuildCount = 0;
	
	public Sentry_CFactory(String IDNumber) {
		super(IDNumber, "Gamma Droids");
	}
	
	public TreeMap<String, SentryDroid_C> buildDroids_C(int count){
		TreeMap<String, SentryDroid_C> Cmap = new TreeMap<String, SentryDroid_C>();
		
		for(int x = 0; x < count; x ++) {
			BuildCount += 1;
			Cmap.put("C" + BuildCount, new SentryDroid_C("C" + BuildCount));
			
		}
		
		return Cmap;
	}

}
