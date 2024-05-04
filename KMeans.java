import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;

import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

public class KMeans extends Configured implements Tool {
    public static int Iterate(Configuration conf) throws IOException {
        FileSystem fs = FileSystem.get(conf);
        Path initialCentroidFile = new Path("/input/centroids.txt");
        Path newCentroidFile = new Path("/output/task_2_1.clusters/part-r-00000");

        if (!(fs.exists(initialCentroidFile) && fs.exists(newCentroidFile)))
            return 1;

        return fs.delete(initialCentroidFile, true) && fs.rename(newCentroidFile, initialCentroidFile) ? 0 : 1;
    }

    public static class KMeansMapper extends Mapper<Object, Text, IntWritable, Text> {
	    public static List<Double[]> centroids = new ArrayList<>();

	    public void setup(Context context) throws IOException { 	
			try {
				Path[] cache = DistributedCache.getLocalCacheFiles(context.getConfiguration());
				if(cache == null || cache.length <= 0) {
					System.exit(1);
				}
				BufferedReader reader = new BufferedReader(new FileReader(cache[0].toString()));
				String line;
				
				while((line = reader.readLine()) != null) {
					String[] str = line.substring(line.indexOf(" ") + 1).split(" ");
					Double[] centroid = new Double[2];
					centroid[0] = Double.parseDouble(str[0]);
					centroid[1] = Double.parseDouble(str[1]);
					centroids.add(centroid);
				}
				reader.close();
			}
			catch(Exception e) {
			}
		}
	    
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			String[] tokens = value.toString().split(" ");
	        double x = Double.parseDouble(tokens[0]);
	        double y = Double.parseDouble(tokens[1]);

			double minDistance = Double.MAX_VALUE;
			int nearestCentroidId = 0;  
			for (int i = 0; i < centroids.size(); i++) {				
				double distance = Math.sqrt(Math.pow(x - centroids.get(i)[0], 2) + Math.pow(y - centroids.get(i)[1], 2));
				if (distance < minDistance) {
					nearestCentroidId = i;
					minDistance = distance;
				}
			}
			context.write(new IntWritable(nearestCentroidId), value);
		}
	}
            
    public static class KMeansReducer extends Reducer<IntWritable, Text, Text, Text> {
        public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            double sumX = 0.0;
			double sumY = 0.0;
            int count = 0;
    
            for (Text value : values) {
                String[] tokens = value.toString().split("\\s+");
                sumX += Double.parseDouble(tokens[0]);
                sumY += Double.parseDouble(tokens[1]);
                count++;
            }
    
            String centroid = sumX / count + " " + sumY / count;
            String clusterId = "Cluster:" + key.get();
            
            context.write(new Text(clusterId), new Text(centroid));
        }
    }
    
    public static class ClusterAssignReducer extends Reducer<IntWritable, Text, Text, Text> {
		public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            for (Text value : values) {
				String clusterId = "Cluster:" + key;
                context.write(new Text(clusterId), value);
            }
        }
    }

    public int run(String[] args) throws Exception {
		Configuration conf = getConf();
		FileSystem fs = FileSystem.get(conf);
		Job job = new Job(conf);
		job.setJarByClass(KMeans.class);
		
		FileInputFormat.setInputPaths(job, "/input/points.txt");
		Path clusterPath = new Path("/output/task_2_1.clusters");
		fs.delete(clusterPath,true);
		FileOutputFormat.setOutputPath(job, clusterPath);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);		
		job.setMapperClass(KMeansMapper.class);
		job.setReducerClass(KMeansReducer.class);		
		job.setNumReduceTasks(1);		
		job.setMapOutputKeyClass(IntWritable.class);
	    job.setMapOutputValueClass(Text.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);	 
		return job.waitForCompletion(true)?0:1;
	}

	public static void RunClass(Configuration conf, FileSystem fs) throws Exception {
		Job job = new Job(conf);
		job.setJarByClass(KMeans.class);
	
		FileInputFormat.setInputPaths(job, "/input/points.txt");
		Path classPath = new Path("/output/task_2_1.classes");
		fs.delete(classPath, true);
		FileOutputFormat.setOutputPath(job, classPath);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		job.setMapperClass(KMeansMapper.class);
		job.setReducerClass(ClusterAssignReducer.class);
		job.setNumReduceTasks(1);
		job.setMapOutputKeyClass(IntWritable.class);
		job.setMapOutputValueClass(Text.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		job.waitForCompletion(true);
	}


	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(conf);

		Path dataFile = new Path("/input/centroids.txt");
		DistributedCache.addCacheFile(dataFile.toUri(), conf);

		int success = 1;
		final int iterations = 20;

		for (int iteration = 1; success == 1 && iteration < iterations; iteration++) {
			if (iteration > 1 && Iterate(conf) != 0) {
				System.exit(1);
			}
			success ^= ToolRunner.run(conf, new KMeans(), args);
		}

		RunClass(conf, fs);
	}
}
