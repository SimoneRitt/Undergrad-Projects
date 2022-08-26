package rittenhouse.FACTORY;

import java.util.TreeMap;

import rittenhouse.DROID.SentryDroid_A;

public class Sentry_AFactory extends DroidFactory {
	
	public static int BuildCount = 0;
	
	public Sentry_AFactory(String IDNumber) {
		super(IDNumber, "Alpha Droids");
	}
	
	public TreeMap<String, SentryDroid_A> buildDroids_A(int count){
		TreeMap<String, SentryDroid_A> Amap = new TreeMap<String, SentryDroid_A>();
		
		for(int x = 0; x < count; x ++) {
			BuildCount += 1;
			Amap.put("A" + BuildCount, new SentryDroid_A("A" + BuildCount));
			
		}
		
		return Amap;
	}

}
