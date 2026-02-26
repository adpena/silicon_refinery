cask "silicon-refinery-chat" do
  version "0.0.209"
  sha256 "3b556f906aba5a65f910068560e9fd04438afb203359071db24e52091999345c"

  url "https://github.com/adpena/silicon-refinery-chat/releases/download/v#{version}/SiliconRefineryChat-#{version}.dmg"
  name "SiliconRefineryChat"
  desc "Standalone macOS app for local Apple Foundation Models chat"
  homepage "https://github.com/adpena/silicon-refinery-chat"

  app "SiliconRefineryChat.app"
  binary "#{appdir}/SiliconRefineryChat.app/Contents/MacOS/SiliconRefineryChat",
         target: "silicon-refinery-chat"

  zap trash: [
    "~/Library/Application Support/com.siliconrefinery.chat",
    "~/Library/Preferences/com.siliconrefinery.chat.plist",
  ]
end
