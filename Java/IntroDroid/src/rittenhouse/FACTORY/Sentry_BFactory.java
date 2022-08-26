package rittenhouse.FACTORY;

import java.util.TreeMap;

import rittenhouse.DROID.SentryDroid_B;

public class Sentry_BFactory extends DroidFactory {

	public static int BuildCount = 0;
	
	public Sentry_BFactory(String IDNumber) {
		super(IDNumber, "Beta Droids");
	}
	
	public TreeMap<String, SentryDroid_B> buildDroids_B(int count){
		TreeMap<String, SentryDroid_B> Bmap = new TreeMap<String, SentryDroid_B>();
		
		for(int x = 0; x < count; x ++) {
			BuildCount += 1;
			Bmap.put("B" + BuildCount, new SentryDroid_B("B" + BuildCount));
			
		}
		
		return Bmap;
	}
}
