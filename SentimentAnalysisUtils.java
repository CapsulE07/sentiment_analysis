
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;

public class SentimentAnalysisUtils {

//    private static final transient Logger logger = LoggerFactory.getLogger(SentimentAnalysisUtils.class);

    /**
     * 远程服务器服务名、地址, 参数
     */
    final static String inputParamsString = "pre_string";
	final static String sentimentAnalysisServerUrl ="http://192.168.52.222:7330/predict";

    public static String getRes(String input) {

        // 输入字符串处理成字典
        Map<String, String> params = new HashMap<String, String>();
        // 访问远程服务器
        params.put(inputParamsString, input);
        String res = "";
        try {
            res = WebUtils.doPost(sentimentAnalysisServerUrl, params, 60000, 60000);

            System.out.println("该输入语句的情感色彩为"+ res);
        } catch (Throwable throwable) {
            System.out.println("情感分析出现异常！");
        }
        return res;
    }


    public static void main(String args[]) throws IOException {

        String res = getRes("光明美丽正义");
    }
}
